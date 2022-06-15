from __future__ import print_function
import configparser
import grpc
import utils.databus.databus_pb2 as dbus_pb2
import utils.databus.databus_pb2_grpc as dbus_grpc
from google.protobuf.json_format import MessageToJson
from utils.utils import *

logger = get_logger(__name__, level='DEBUG')



class Client(object):
    def __init__(self) -> None:
        config = configparser.ConfigParser()
        config.read('client.conf')
        
        self.metainfo = config['meta']
        self.channel = grpc.secure_channel('{}:{}'.format(self.metainfo.server, self.metainfo.server_port))
        self.connection_stub = dbus_grpc.ConnectionStub(self.channel)
        self.registration_stub = dbus_grpc.RegistrationStub(self.channel)
        self.heartbeat_stub = dbus_grpc.HeartbeatStub(self.channel)
        self.cachemiss_stub = dbus_grpc.CacheMissStub(self.channel)

        conn_req = dbus_pb2.ConnectRequest(cred=dbus_pb2.Credential(self.metainfo.usermame, self.metainfo.password), createUser=True)
        resp = self.connection_stub.connect(conn_req)
        if resp.rc == dbus_pb2.FAILED:
            logger.error('Failed to connect to server with error messsge {}'.format(resp.resp))
            raise Exception(resp.resp)
        else:
            logger.info('Connect to server')
        
        self.jobs = {}
    
    def register_jobs(self, specs):
        """Register a list of jobs to the GM

        Args:
            spec (array): a list of job specifications
        """        
        for job_spec in specs:
            ds = job_spec.datasource
            request = dbus_pb2.RegisterRequest(
                cred=dbus_pb2.Credential(username=self.metainfo.username, password=self.metainfo.password, createUser=True),
                dataset=ds.name,
                s3auth=dbus_pb2.S3Auth(ds.aws_access_key_id, ds.aws_secret_access_key, ds.region_name, ds.bucket, ds.keys),
                resource=dbus_pb2.ResourceInfo(CPUMemoryFree=get_cpu_free_mem(), CPUMemoryFree=get_gpu_free_mem())
            )
            resp = self.registration_stub.register(request)
            if type(resp) is dbus_pb2.RegisterSuccess:
                self.jobs[job_spec.jobId] = resp
                logger.info('Registered job {}, assigned jobId is {}'.format(job_spec.name, resp.jinfo.jobId))
            else:
                logger.error('Failed to register job {}, due to {}'.format(job_spec.name, resp.error))
        
    

if __name__ == '__main__':
    client = Client()