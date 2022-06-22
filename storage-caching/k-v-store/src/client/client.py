from __future__ import print_function
from concurrent.futures import ThreadPoolExecutor
import grpc
import signal
import json
import time
import configparser
import grpctool.dbus_pb2 as pb
import grpctool.dbus_pb2_grpc as pb_grpc
from google.protobuf.json_format import MessageToJson
from utils import *
from watchdog.observers import Observer
from watchdog.events import *


logger = get_logger(__name__, level='Info')
HEARTBEAT_FREQ = 10



class Client(FileSystemEventHandler):
    def __init__(self) -> None:
        secret = configparser.ConfigParser()
        secret.read("/config/client.conf")
        self.manager_sec = secret['alnair_manager']
        aws_s3_sec = secret["aws_s3"]
        
        self.jobs = configparser.ConfigParser()
        self.jobs.read("/config/client.conf")
        
        self.channel = grpc.secure_channel('{}:{}'.format(self.manager_sec["server_address"], self.manager_sec["server_port"]))
        self.connection_stub = pb_grpc.ConnectionStub(self.channel)
        req = pb.ConnectRequest(
            cred=pb.Credential(self.manager_sec["username"], self.manager_sec["password"]), 
            s3auth=pb.S3Auth(aws_s3_sec["aws_access_key_id"], aws_s3_sec["aws_secret_access_key"], aws_s3_sec["region_name"]),
            createUser=True
        )
        resp = self.connection_stub.connect(req)
        if resp.rc == pb.FAILED:
            logger.error("failed to connect to server with: {}".format(resp.resp))
            raise Exception
        else:
            logger.info("connect to server")
        
        self.running_jobs = {}
        self.registration_stub = pb_grpc.RegistrationStub(self.channel)
        self.register_jobs()
        
        self.heartbeat_stub = pb_grpc.HeartbeatStub(self.channel)
        self.hb_pool = ThreadPoolExecutor(max_workers=len(self.jobs))
        for _, job in self.jobs.items():
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
            qos = job['QoS']
            request = pb.RegisterRequest(
                cred=pb.Credential(username=self.manager_sec["username"], password=self.manager_sec["password"], createUser=True),
                datasource=pb.DataSource(name=job['name'], bucket=job['bucket'], keys=job['keys']),
                qos=pb.QoS(
                    useCache=qos['useCache'],
                    flushFreq=qos['flushFreq'],
                    durabilityInMem=qos['durabilityInMem'],
                    durabilityInDisk=qos['durabilityInDisk'],
                ),
                resource=pb.ResourceInfo(CPUMemoryFree=get_cpu_free_mem(), GPUMemoryFree=get_gpu_free_mem())
            )
            resp = self.registration_stub.register(request)
            if type(resp) is pb.RegisterSuccess:
                self.running_jobs[job['name']] = resp
                logger.info('registered job {}, assigned jobId is {}'.format(job['name'], resp.jinfo.jobId))
                with open('/data/{}.json'.format(job['name']), 'w') as f:
                    json.dump(MessageToJson(resp), f)
            else:
                logger.error('failed to register job {}, due to {}'.format(job['name'], resp.error))
                os.kill(os.getpid(), signal.SIGINT)
    
    def send_hearbeat(self, job: pb.JobInfo):
        """client sends hearbeat to GM to gather latest token

        Args:
            job (_type_): JobInfo object
        """
        while True:
            logger.info("send heartbeat")
            hb = pb.HearbeatMessage(job)
            resp = self.heartbeat_stub.call(hb)
            if resp.jinfo.token != job.token:
                logger.info("update token for job {}".format(job.name))
                self.jobs[job.name] = resp
                with open('/data/{}.json'.format(job.name), 'w') as f:
                    json.dump(MessageToJson(resp), f)
            time.sleep(hb)
    
    def handle_cachemiss(self):
        with open('/data/cachemiss', 'r') as f:
            misskeys = f.readlines()
        for key in misskeys:
            resp = self.cachemiss_stub.call(pb.CacheMissRequest(key))
            if resp.response:
                logger.info('request missing key {}'.format(key))
            else:
                logger.warning('failed to request missing key {}'.format(key))
    
    def on_modified(self, event):
        if event.src_path == '/share/cachemiss':
            return self.handle_cachemiss()
    
    def prob_job(self, job: pb.JobInfo):
        # TODO: probe job runtime information
        pass

if __name__ == '__main__':
    client = Client()
    fs_observer = Observer()
    fs_observer.schedule(client, r"/share/cachemiss", True)
    fs_observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        fs_observer.stop()