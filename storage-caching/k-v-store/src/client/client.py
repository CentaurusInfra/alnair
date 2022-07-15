from __future__ import print_function
from concurrent.futures import ThreadPoolExecutor
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
from watchdog.observers import Observer
from watchdog.events import *
from utils import *


logger = get_logger(__name__, level='Info')
HEARTBEAT_FREQ = 10


class Client(FileSystemEventHandler):
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
    
    def handle_cachemiss(self):
        with open('/share/cachemiss', 'r') as f:
            misskeys = f.readlines()
        for key in misskeys:
            resp = self.cm_stub.call(pb.CacheMissRequest(cred=self.cred, key=key))
            if resp.response:
                logger.info('request missing key {}'.format(key))
            else:
                logger.warning('failed to request missing key {}'.format(key))
    
    def on_created(self, event):
        if event.src_path == '/share/cachemiss':
            return self.handle_cachemiss()
        
    def on_modified(self, event):
        if event.src_path == '/share/cachemiss':
            return self.handle_cachemiss()
    
    def prob_job(self, job: pb.JobInfo):
        # TODO: probe job runtime information
        pass

if __name__ == '__main__':
    client = Client()
    Path("/share/cachemiss").touch()
    fs_observer = Observer()
    fs_observer.schedule(client, "/share/cachemiss")
    fs_observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        fs_observer.stop()