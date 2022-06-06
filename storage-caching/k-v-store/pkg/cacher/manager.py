from concurrent import futures
from email import message

import grpc

import utils.databus.databus_pb2 as dbus_pb2
import utils.databus.databus_pb2_grpc as dbus_grpc


class Connection(dbus_grpc.ConnectionServicer):
    def Connect(self, request, context):
        """
        TODO:
        1. check authorization information
        2. save to db
        """
        return dbus_pb2.ConnectResponse(

        )
    

class Registration(dbus_grpc.RegistrationServicer):
    def Register(self, request, context):
        """
        TODO: 
        1. save job to db
        2. get dataset meta-info from S3
        3. generate initial caching policy
        """
        return dbus_pb2.RegisterResponse(
            
        )
    
    def Deresgister(self, request, context):
        """
        TODO: 
        1. delete job from db
        2. evict data based on user specification
        """
        return dbus_pb2.DeregisterResponse(
            
        )
    

class UpdatePolicy(dbus_grpc.UpdatePolicyServicer):
    def Update(self, request, context):
        return super().Update(request, context)


class Logger(dbus_grpc.LoggerServicer):
    def call(self, request, context):
        return super().call(request, context)