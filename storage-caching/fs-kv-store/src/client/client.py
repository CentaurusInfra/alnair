from __future__ import print_function
from os import environ
import grpc
import signal
import json
import time
import glob
from pathlib import Path
import configparser
import grpctool.dbus_pb2 as pb
import grpctool.dbus_pb2_grpc as pb_grpc
from google.protobuf.json_format import ParseDict
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
        
        self.jobsmeta = []
        for f in glob.glob('/jobsmeta/*.json'):
            with open(f, 'rb') as f:
                job = json.load(f)
            if job['qos']['UseCache']:
                self.jobsmeta.append(job)
        
        if len(self.jobsmeta) > 0:
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
            
            self.register_stub = pb_grpc.RegistrationStub(self.channel)
            self.register_job()
            self.datamiss_stub = pb_grpc.DataMissStub(self.channel)
                        
            self.runtime_buffer = OrderedDict()
            self.req_time = []
            self.load_time = []
            
            # runtime tmpfs waterline: n*num_workers*batch_size, n=2 initially
            self.waterline = 2
            self.pidx = 0
            self.pf_paths = None
            
            signal.signal(signal.SIGINT, self.exit_gracefully)
            signal.signal(signal.SIGTERM, self.exit_gracefully)
            
    def exit_gracefully(self):
        self.channel.close()
            
    def register_job(self):
        """Register a list of jobs to the GM

        Args:
            spec (array): a list of job specifications
        """
        for job in self.jobsmeta:
            qos = job['qos']
            ds = job['datasource']
            if 'keys' not in ds: ds['keys'] = []
            request = pb.RegisterRequest(
                cred=self.cred,
                nodeIP=environ.get('NODE_IP'),
                datasource=pb.DataSource(name=ds['name'], bucket=ds['bucket'], keys=ds['keys']),
                nodePriority=job['nodePriority'],
                qos=ParseDict(qos, pb.QoS(), ignore_unknown_fields=True),
                resource=pb.ResourceInfo(CPUMemoryFree=get_cpu_free_mem(), GPUMemoryFree=get_gpu_free_mem())
            )
            logger.info('waiting for data preparation')
            resp = self.register_stub.register(request)
            logger.info('receiving registration response stream')
            if resp.rc == pb.RC.REGISTERED:
                resp = resp.regsucc
            else:
                resp = resp.regerr
                logger.error("failed to register job {}: {}".format(job['name'], resp.error))
                os.kill(os.getpid(), signal.SIGINT)
            logger.info('registered job {}, assigned jobId is {}'.format(job['name'], resp.jinfo.jobId))
            while not self.pf_paths: pass
            for _ in range(self.waterline): 
                self.prefetch()
            with open('/share/{}.json'.format(job['name']), 'w') as f:
                json.dump(MessageToDict(resp), f)

    def handle_datamiss(self):
        with open('/share/datamiss', 'r') as f:
            misskeys = f.readlines()
        for key in misskeys:
            resp = self.datamiss_stub.call(pb.DataMissRequest(cred=self.cred, key=key))
            if resp.response:
                logger.info('request missing key {}'.format(key))
            else:
                logger.warning('failed to request missing key {}'.format(key))

    def prefetch(self):
        if self.runtime_conf['LazyLoading']:
            for _ in range(self.runtime_conf['num_workers']):
                for path in self.pf_paths[self.pidx]:
                    if self.pidx not in self.runtime_buffer:
                        self.runtime_buffer[self.pidx] = []
                    if path not in self.runtime_buffer[self.pidx]:
                        shutil.copyfile(path, '/runtime/{}'.format(path))  # NFS --> tmpfs
                        self.runtime_buffer[self.pidx].append(path)
                self.pidx += 1
        else:
            path = self.pf_paths[self.pidx]
            shutil.copyfile(path, '/runtime/{}'.format(path))  # NFS --> tmpfs
            self.pidx += 1

    def process_IN_CREATE(self, event):
        if event.pathname == '/share/datamiss':
            return self.handle_datamiss()
        elif event.pathname == '/share/next':
            self.req_time.append(time.time())
            self.prefetch()
        elif event.path == '/share/prefetch_policy.json':
            with open('/share/prefetch_policy.json', 'r') as f:
                tmp = json.load(f)
                self.runtime_conf = tmp['meta']
                self.pf_paths = tmp['policy']

    def process_IN_MODIFY(self, event):
        self.process_IN_CREATE(event)
            
    def process_IN_CLOSE_NOWRITE(self, event):
        if '/runtime' in event.pathname:
            # pop head
            batch_index = self.runtime_buffer.keys[0]
            self.runtime_buffer[batch_index].pop(0)
            if len(self.runtime_buffer[batch_index]) == 0:
                self.runtime_buffer.popitem(last=False)
            shutil.rmtree(event.pathname, ignore_errors=True)
            
            # tune buffer size
            if len(self.req_time) > 1:
                # decide alpha and beta based on the latest 3 measurements
                alpha = np.diff(self.req_time[-4:])[1:]
                beta = np.array(self.load_time[-3:])
                N = len(self.pf_paths)
                k = len(self.req_time)
                """
                To ensure the data is always available for DataLoader, the length of buffer should be:
                s >= 2*B, if alpha >= beta; otherwise,
                s >= (N-k)*(1-alpha/beta) 
                """
                s = max(2, np.mean((1-alpha/beta)*(N-k), dtype=int))
                
                # update waterline according to load/consume speed
                if self.waterline == s:
                    return
                else:
                    self.waterline = s
                    while len(self.runtime_buffer) > s:
                        self.runtime_buffer.popitem(last=False)
                        self.pidx -= 1
                    while len(self.runtime_buffer) < s:
                        self.prefetch()


if __name__ == '__main__':
    client = Client()
    if not os.path.exists("/share/datamiss"):
        Path("/share/datamiss").touch()
    wm = pyinotify.WatchManager()
    mask = pyinotify.IN_CREATE | pyinotify.IN_MODIFY | pyinotify.IN_CLOSE_NOWRITE
    wm.add_watch("/share", mask)
    wm.add_watch('/runtime', mask)
    notifier = pyinotify.Notifier(wm, client)
    try:
        notifier.loop()
    except KeyboardInterrupt:
        notifier.stop()