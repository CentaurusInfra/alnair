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
from google.protobuf.json_format import MessageToDict, ParseDict
from watchdog.observers import Observer
from watchdog.events import *
from utils import *


logger = get_logger(__name__, level='Info')
HEARTBEAT_FREQ = 10


class Client(FileSystemEventHandler):
    def __init__(self):
        secret = configparser.ConfigParser()
        secret.read("/secret/client.conf")
        manager_sec = secret['alnair_manager']
        
        aws_s3_sec = dict(secret["aws_s3"].items())
        self.cred = pb.Credential(username=manager_sec["username"], password=manager_sec["password"])
        
        self.jobs = []
        for f in glob.glob('/jobs/*.job'):
            with open(f, 'rb') as f:
                job = json.load(f)
            self.jobs.append(job)
        
        self.channel = grpc.insecure_channel('{}:{}'.format(manager_sec["server_address"], manager_sec["server_port"]))
        self.connection_stub = pb_grpc.ConnectionStub(self.channel)
        
        req = pb.ConnectRequest(
            cred=self.cred, 
            s3auth=ParseDict(aws_s3_sec, pb.S3Auth(), ignore_unknown_fields=True),
            createUser=True
        )
        resp = self.connection_stub.connect(req)
        if resp.rc == pb.RC.FAILED:
            logger.error("failed to connect to server with: {}".format(resp.resp))
            raise Exception
        else:
            logger.info("connect to server")
        
        self.registered_jobs = {}
        self.registration_stub = pb_grpc.RegistrationStub(self.channel)
        self.register_jobs()
        
        self.heartbeat_stub = pb_grpc.HeartbeatStub(self.channel)
        self.hb_pool = ThreadPoolExecutor(max_workers=len(self.registered_jobs))
        for _, job in self.registered_jobs.items():
            self.hb_pool.submit(self.send_hearbeat, job)
            
        self.cachemiss_stub = pb_grpc.CacheMissStub(self.channel)
        
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self):
        self.hb_pool.shutdown(wait=False)
        self.channel.close()
    
    def register_jobs(self):
        """Register a list of jobs to the GM

        Args:
            spec (array): a list of job specifications
        """
        for job in self.jobs:
            qos = job['qos']
            ds = job['datasource']
            request = pb.RegisterRequest(
                cred=self.cred,
                datasource=pb.DataSource(name=ds['name'], bucket=ds['bucket'], keys=ds['keys']),
                qos=ParseDict(qos, pb.QoS(), ignore_unknown_fields=True),
                resource=pb.ResourceInfo(CPUMemoryFree=get_cpu_free_mem(), GPUMemoryFree=get_gpu_free_mem())
            )
            resp = self.registration_stub.register(request)
            if resp.rc == pb.RC.REGISTERED:
                resp = resp.regsucc
                self.registered_jobs[job['name']] = resp.jinfo
                logger.info('registered job {}, assigned jobId is {}'.format(job['name'], resp.jinfo.jobId))
                with open('/share/{}.json'.format(job['name']), 'w') as f:
                    json.dump(MessageToDict(resp), f)
            else:
                resp = resp.regerr
                logger.error("failed to register job {}: {}".format(job['name'], resp.error))
                os.kill(os.getpid(), signal.SIGINT)
    
    def send_hearbeat(self, job: pb.JobInfo):
        """client sends hearbeat to GM to gather latest token

        Args:
            job (_type_): JobInfo object
        """
        while True:
            logger.info("send heartbeat")
            hb = pb.HearbeatMessage(cred=self.cred, jinfo=job)
            resp = self.heartbeat_stub.call(hb)
            if resp.jinfo.token != job.token:
                logger.info("update token for job {}".format(job.name))
                self.registered_jobs[job.name] = resp
                with open('/share/{}.json'.format(job.name), 'w') as f:
                    json.dump(MessageToDict(resp), f)
            time.sleep(hb)
    
    def handle_cachemiss(self):
        with open('/share/cachemiss', 'r') as f:
            misskeys = f.readlines()
        for key in misskeys:
            resp = self.cachemiss_stub.call(pb.CacheMissRequest(cred=self.cred, key=key))
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