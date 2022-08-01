from __future__ import print_function
from concurrent.futures import ThreadPoolExecutor
from os import environ
import grpc
import signal
import json
import multiprocessing
import time
import glob
from pathlib import Path
import configparser
import grpctool.dbus_pb2 as pb
import grpctool.dbus_pb2_grpc as pb_grpc
from google.protobuf.json_format import ParseDict
import concurrent
import pyinotify
import shutil
from collections import OrderedDict
import numpy as np
from utils import *


logger = get_logger(__name__, level='Info')
HEARTBEAT_FREQ = 10


class Client(pyinotify.ProcessEvent):
    def __init__(self):
        secret = configparser.ConfigParser()
        secret.read("/secret/client.conf")
        aws_s3_sec = dict(secret["aws_s3"].items())
        
        self.rj = [] # jobs using cache cluster
        self.sj = [] # jobs using s3
        for f in glob.glob('/jobs/*.json'):
            with open(f, 'rb') as f:
                job = json.load(f)
            if job['qos']['UseCache']:
                self.rj.append(job)
            else:
                self.sj.append(job)
        
        if len(self.rj) > 0:
            manager_sec = secret['alnair_manager']
            self.cred = pb.Credential(username=manager_sec["username"], password=manager_sec["password"])
            self.channel = grpc.insecure_channel('{}:{}'.format(manager_sec["server_address"], manager_sec["server_port"]))
            self.conn_stub = pb_grpc.ConnectionStub(self.channel)
            
            req = pb.ConnectRequest(
                cred=self.cred, 
                s3auth=ParseDict(aws_s3_sec, pb.S3Auth(), ignore_unknown_fields=True),
                createUser=True
            )
            resp = self.conn_stub.connect(req)
            if resp.rc == pb.RC.FAILED:
                logger.error("failed to connect to server with: {}".format(resp.resp))
                raise Exception
            else:
                logger.info("connect to server")
            
            self.reg_rj = {}
            self.reg_stub = pb_grpc.RegistrationStub(self.channel)
            self.register_rj()
            
            self.hb_stub = pb_grpc.HeartbeatStub(self.channel)
            self.hb_pool = ThreadPoolExecutor(max_workers=len(self.reg_rj))
            for _, job in self.reg_rj.items():
                self.hb_pool.submit(self.send_hearbeat, job)
                
            self.cm_stub = pb_grpc.CacheMissStub(self.channel)
            
            signal.signal(signal.SIGINT, self.exit_gracefully)
            signal.signal(signal.SIGTERM, self.exit_gracefully)
            
            self.runtime_buffer = OrderedDict()
            self.req_time = []
            self.load_time = []
            self.wl = 2
            self.batch_index = 0
            self.batch_indices = None
            
    def exit_gracefully(self):
        self.hb_pool.shutdown(wait=False)
        self.channel.close()
            
    def register_rj(self):
        """Register a list of jobs to the GM

        Args:
            spec (array): a list of job specifications
        """
        for job in self.rj:
            qos = job['qos']
            ds = job['datasource']
            if 'keys' not in ds: ds['keys'] = []
            request = pb.RegisterRequest(
                cred=self.cred,
                nodeIP=environ.get('NODE_IP'),
                datasource=pb.DataSource(name=ds['name'], bucket=ds['bucket'], keys=ds['keys']),
                qos=ParseDict(qos, pb.QoS(), ignore_unknown_fields=True),
                resource=pb.ResourceInfo(CPUMemoryFree=get_cpu_free_mem(), GPUMemoryFree=get_gpu_free_mem())
            )
            logger.info('waiting for data preparation')
            resp = self.reg_stub.register(request)
            logger.info('receiving registration response stream')
            if resp.rc == pb.RC.REGISTERED:
                resp = resp.regsucc
            else:
                resp = resp.regerr
                logger.error("failed to register job {}: {}".format(job['name'], resp.error))
                os.kill(os.getpid(), signal.SIGINT)
            self.reg_rj[job['name']] = resp.jinfo
            logger.info('registered job {}, assigned jobId is {}'.format(job['name'], resp.jinfo.jobId))
            while self.batch_indices is None: pass
            for _ in range(self.wl): 
                self.push_data()
            with open('/share/{}.json'.format(job['name']), 'w') as f:
                json.dump(MessageToDict(resp), f)
                        
    def send_hearbeat(self, job: pb.JobInfo):
        """client sends hearbeat to GM to gather latest token

        Args:
            job (_type_): JobInfo object
        """
        while True:
            logger.info("send heartbeat")
            hb = pb.HearbeatMessage(cred=self.cred, jinfo=job)
            resp = self.hb_stub.call(hb)
            if resp.jinfo.token != job.token:
                logger.info("update token for job {}".format(job.name))
                self.reg_rj[job.name] = resp
                with open('/share/{}.json'.format(job.name), 'w') as f:
                    json.dump(MessageToDict(resp), f)
            time.sleep(hb)
    
    # TODO: 1. handle cache miss in tmpfs
    # 2. handle cache miss in NFS
    def handle_cachemiss(self):
        with open('/share/cachemiss', 'r') as f:
            misskeys = f.readlines()
        for key in misskeys:
            resp = self.cm_stub.call(pb.CacheMissRequest(cred=self.cred, key=key))
            if resp.response:
                logger.info('request missing key {}'.format(key))
            else:
                logger.warning('failed to request missing key {}'.format(key))
    
    # TODO: check how to push missed key into tmpfs
    def push_data(self, keys=None):
        def cpy(key, batch_index):
            if batch_index not in self.runtime_buffer:
                self.runtime_buffer[batch_index] = []
            if key not in self.runtime_buffer[batch_index]:
                shutil.copyfile(key, '/runtime/{}'.format(key.split('/')[1]))
                self.runtime_buffer[batch_index].append(key)
        with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            futures = []
            if keys is None:
                for _ in self.runtime_conf['num_workers']:
                    for key in self.batch_indices[self.batch_index]:
                        futures.append(executor.submit(cpy, key, self.batch_index))
                    self.batch_index += 1
            else:
                for key in keys:
                    futures.append(executor.submit(cpy, key, 'missing'))
        concurrent.futures.wait(futures)

    def process_IN_CREATE(self, event):
        if event.pathname == '/share/cachemiss':
            return self.handle_cachemiss()
        elif event.pathname == '/share/runtime_conf.json':
            with open('/share/runtime_conf.json', 'r') as f:
                self.runtime_conf = json.load(f)
        elif event.pathname == '/share/nextbatch':
            self.req_time.append(time.time())
            self.push_data()
        elif event.path == '/share/batch_indices.npy':
            self.batch_indices = np.load('/share/batch_indices.npy')

    def process_IN_MODIFY(self, event):
        self.process_IN_CREATE(event)
            
    def process_IN_CLOSE_NOWRITE(self, event):
        if '/runtime' in event.pathname:
            h = self.runtime_buffer.keys[0]
            self.runtime_buffer[h].pop(0)
            if len(self.runtime_buffer[h]) == 0:
                self.runtime_buffer.popitem(last=False)
            try:
                shutil.rmtree(event.pathname, ignore_errors=False)
            except Exception as ex:
                logger.error("failed to delete file {}: {}".format(event.pathname, str(ex)))
            
            # adjust buffer size
            if len(self.req_time) > 1:
                # decide alpha and beta based on the latest 3 measurements
                alpha = np.diff(self.req_time[-4:])[1:]
                beta = np.array(self.load_time[-3:])
                N = len(self.batch_indices)
                k = len(self.req_time)
                """
                To ensure the data is always available for DataLoader, the length of buffer should be:
                s >= 2*B, if alpha >= beta; otherwise,
                s >= (N-k)*(1-alpha/beta) 
                """
                s = max(2, np.mean((1-alpha/beta)*(N-k), dtype=int))
                if self.wl == s: 
                    return
                else:
                    self.wl = s
                    while len(self.runtime_buffer) > s:
                        self.runtime_buffer.popitem(last=False)
                        self.batch_index -= 1
                    while len(self.runtime_buffer) < s:
                        self.push_data()


if __name__ == '__main__':
    client = Client()
    if not os.path.exists("/share/cachemiss"):
        Path("/share/cachemiss").touch()
    wm = pyinotify.WatchManager()
    mask = pyinotify.IN_CREATE | pyinotify.IN_MODIFY | pyinotify.IN_CLOSE_NOWRITE
    wm.add_watch("/share", mask)
    wm.add_watch('/runtime', mask)
    notifier = pyinotify.Notifier(wm, client)
    try:
        notifier.loop()
    except KeyboardInterrupt:
        notifier.stop()