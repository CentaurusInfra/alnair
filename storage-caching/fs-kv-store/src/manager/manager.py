import concurrent.futures
import sys
import shutil
import grpc
import boto3
import threading
import json, bson
import time
import configparser
import multiprocessing
import grpctool.dbus_pb2 as pb
import grpctool.dbus_pb2_grpc as pb_grpc
from datetime import datetime
from pymongo.mongo_client import MongoClient
from pytz import timezone
import glob
from utils import *


logger = get_logger(name=__name__, level='debug')


class Manager(object):
    def __init__(self):
        parser = configparser.ConfigParser()
        parser.read('/configs/manager/manager.conf')
        
        try:
            self.managerconf = parser['manager']
            mconf = parser['mongodb']
        except KeyError as err:
            logger.error(err)
        
        mongo_client = MongoClient(host=mconf['host'], port=int(mconf['port']), username=mconf['username'], password=mconf['password'])
        with open("mongo-schemas/client.json", 'r') as f:
            client_schema = json.load(f)
        with open("mongo-schemas/job.json", 'r') as f:
            job_schema = json.load(f)
        
        # mongo_client.drop_database("Cacher")
        
        collections = mongo_client.Cacher.list_collection_names()
        if 'Client' not in collections:
            self.client_col = mongo_client.Cacher.create_collection(name="Client", validator={"$jsonSchema": client_schema}, validationAction="error")
        else:
            self.client_col = mongo_client.Cacher.Client
            
        if 'Job' not in collections: 
            self.job_col = mongo_client.Cacher.create_collection(name='Job', validator={"$jsonSchema": job_schema}, validationAction="error")
        else:
            self.job_col = mongo_client.Cacher.Job
        
        flush_thrd = threading.Thread(target=Manager.flush_data, args=(self,), daemon=True)
        flush_thrd.start()
        
        logger.info("start global manager")

    def auth_client(self, cred, s3auth=None, conn_check=False):
        result = self.client_col.find_one(filter={"username": cred.username})
        if result is None:
                rc = pb.RC.NO_USER
        else:
            if cred.password == result['password']:
                if conn_check:
                    rc = pb.RC.CONNECTED if result['status'] else pb.RC.DISCONNECTED
                else:
                    rc = pb.RC.CONNECTED
            else:
                rc = pb.RC.WRONG_PASSWORD
        # check whether to update s3auth information
        if rc == pb.RC.CONNECTED and s3auth is not None and result['s3auth'] != s3auth:
            result = self.client_col.update_one(
                filter={"username": cred.username}, 
                update={"$set": {"s3auth": s3auth}})
            if result.modified_count != 1:
                logger.error("user {} is connected, but failed to update S3 authorization information.".format(cred.username))
                rc = pb.RC.FAILED
        return rc
    
    def calculate_max_chunk_size(self):
        """
        Returns:
            _type_: chunk size in MB. Defaults to 512MB
        """
        DEFAULT_CHUNK_SIZE = 512*1024*1024
        return DEFAULT_CHUNK_SIZE

    def assign_location(self, bucket_objs, servers, waterline=0.1):
        k = 0
        while k < len(bucket_objs):
            obj = bucket_objs[k]
            flag = False
            for svr in servers:
                total, _, free = shutil.disk_usage(svr)
                if free/total > waterline and free > obj['Size']:
                    bucket_objs[k]['Location'] = svr
                    flag = True
                    break
            if not flag: # NFS out-of-space
                self.evict_data()
            k += 1
        
    def evict_data(self):
        def lru(n=1):
            """Backup the least N recent used keys

            Args:
                n (int, optional): the number of keys. Defaults to 1.
            """
            return [
                {"$unwind": "$policy.chunks"},
                {"$project": {
                    "_id": 0, 
                    "key": "$policy.chunks.Key",
                    "lastAccessTime": "$policy.chunks.LastAccessTime", 
                    "location": "$policy.chunks.Location"}
                 },
                {"$sort": {"LastAccessTime": 1}},
                {"$project": {"Key": 1, "Location": 1}},
                {"$limit": n}
            ]
            
        def lfu(n=1):
            """Backup the least N frequent used keys

            Args:
                n (int, optional): the number of keys. Defaults to 1.
            """
            return [
                {"$unwind": "$policy.chunks"},
                {"$project": {
                    "_id": 0, 
                    "key": "$policy.chunks.Key",
                    "totalAccessTime": "$policy.chunks.TotalAccessTime", 
                    "location": "$policy.chunks.Location"}
                },
                {"$sort": {"TotalAccessTime": 1}},
                {"$project": {"Key": 1, "Location": 1}},
                {"$limit": n}
            ]
        
        pipeline = {"lru": lru, "lfu": lfu}[self.managerconf['eviction-policy']](n=self.managerconf.getint('eviction-size'))
        rmobjs = self.job_col.aggregate(pipeline)
        for obj in rmobjs:
            shutil.rmtree(obj['Location'], ignore_errors=True)
        rmkeys = [obj['Key'] for obj in rmobjs]
        self.job_col.delete_many(
            {"policy.chunks": {"$elemMatch": {"Key": {"$in": rmkeys}}}}
        )


class ConnectionService(pb_grpc.ConnectionServicer):
    def __init__(self, manager: Manager) -> None:
        super().__init__()
        self.manager = manager
        
    def connect(self, request, context):
        cred, s3auth = request.cred, request.s3auth
        rc = self.manager.auth_client(cred=cred, s3auth=MessageToDict(s3auth))
        if rc == pb.RC.WRONG_PASSWORD:
            resp = pb.ConnectResponse(rc=pb.RC.FAILED, resp="wrong password")
        elif rc == pb.RC.NO_USER:
            if request.createUser:
                try:
                    result = self.manager.client_col.insert_one({
                        "username": cred.username,
                        "password": cred.password,
                        "s3auth": MessageToDict(s3auth),
                        "status": True
                    })
                except Exception as ex:
                    print(ex)
                if result.acknowledged:
                    logger.info("user {} connected".format(cred.username))
                    resp = pb.ConnectResponse(rc=pb.RC.CONNECTED, resp="connection setup")
                else:
                    resp = pb.ConnectResponse(rc=pb.RC.FAILED, resp="connection error")
            else:
                resp = pb.ConnectResponse(rc=pb.RC.FAILED, resp = "not found user {}".format(cred.username))
        elif rc == pb.RC.DISCONNECTED:
            result = self.manager.client_col.update_one(
                filter={
                    "username": cred.username,
                    "password": cred.password,
                },
                update={"$set": {"status": True, "jobs": []}}
            )
            if result['modified_count'] == 0:
                resp = pb.ConnectResponse(rc=pb.RC.FAILED, resp="connection error")
            else:
                resp = pb.ConnectResponse(rc=pb.RC.CONNECTED, resp="connection setup")
                logger.info("user {} connected".format(cred.username))
        else:
            resp = pb.ConnectResponse(rc=pb.RC.CONNECTED, resp="connection setup")
            logger.info("user {} connected".format(cred.username))
        return resp


class RegistrationService(pb_grpc.RegistrationServicer):
    def __init__(self, manager: Manager) -> None:
        super().__init__()
        self.manager = manager
    
    def register(self, request, context):
        cred = request.cred
        rc = self.manager.auth_client(cred, conn_check=True)
        jobId = "{}-{}".format(cred.username, request.datasource.name)
        if rc == pb.RC.CONNECTED:
            # get s3 auth
            result = self.manager.client_col.find_one(filter={"$and": [{"username": cred.username, "password": cred.password}]})
            s3auth = result['s3auth']
            s3_session = boto3.Session(
                aws_access_key_id=s3auth['aws_access_key_id'],
                aws_secret_access_key=s3auth['aws_secret_access_key'],
                region_name=s3auth['region_name']
            )
            s3_client = s3_session.client('s3')
            bucket_name = request.datasource.bucket

            # get object keys that are not in MongoDB
            saved_keys = {}
            try:
                saved_job = self.manager.job_col.aggregate([
                    {"$match": {"meta.jobId": jobId}},
                    {"$sort": {"meta.createTime": -1}},
                    {"$limit": 1}
                ]).next()
            except:
                saved_job = None
            if saved_job is not None:
                for chunk in saved_job['policy']['chunks']:
                    saved_keys[chunk['Key']] = chunk
            
            # get bucket objects
            bucket_objs = []
            def list_modified_objects(prefix, page):
                nonlocal bucket_objs
                for info in page['Contents']:
                    info['prefix'] = prefix
                    if saved_job is not None \
                        and info['Key'] in saved_keys \
                        and info['LastModified'].replace(tzinfo=timezone('UTC')).timestamp() == saved_keys[info['Key']]['LastModified']:
                        info['Exist'] = True
                    else:
                        info['Exist'] = False
                    bucket_objs.append(info)
            if len(request.datasource.keys) == 0:
                request.datasource.keys = [bucket_name]
            for prefix in request.datasource.keys:
                paginator = s3_client.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
                futures = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                    for page in pages:
                        futures.append(executor.submit(list_modified_objects, prefix, page))
                concurrent.futures.wait(futures)
                
            # designate node sequence of storing data
            self.manager.assign_location(bucket_objs, request.nodePriority)
            
            # copy data from S3 to NFS
            key_lookup = {}
            def copy_data(s3obj: dict, max_chunk_size: int):
                nonlocal key_lookup
                obj_chunks = []
                
                key = s3obj['Key']
                loc = s3obj['Location']
                size = s3obj['Size']
                lm = s3obj['LastModified']
                
                s3obj['LastModified'] = int(lm.timestamp())
                if not s3obj['Exist']:
                    if size <= max_chunk_size:
                        value = s3_client.get_object(Bucket=bucket_name, Key=key)['Body'].read()
                        hash_key = "/{}/{}".format(loc, hashing(value))
                        with open('/{}/{}'.format(loc, hash_key), 'wb') as f:
                            f.write(value)
                        # logger.info("Copy data from s3:{} to alnair:{}".format(info['Key'], hash_key))
                        s3obj['HashKey'] = hash_key
                        obj_chunks.append(s3obj)
                    else:
                        s3_client.download_file(Bucket=bucket_name, Key=key, Filename='/tmp/{}'.format(key))
                        # logger.info("Download large file s3:{}, size: {}B".format(info['Key'], info['Size']))
                        # TODO: how to ensure the data partition operation does not break data items
                        with open('/tmp/{}'.format(key), 'rb') as f:
                            value = f.read(max_chunk_size)
                            part = 0
                            while value:
                                hash_key = "/{}/{}".format(loc, hashing(value))
                                with open('/{}/{}'.format(loc, hash_key), 'wb') as f:
                                    f.write(value)
                                s3obj['Key'] = '{}.part.{}'.format(s3obj['Key'], part)
                                s3obj['HashKey'] = hash_key
                                s3obj['Size'] = sys.getsizeof(value)
                                obj_chunks.append(s3obj)
                                # logger.info("Copy data from /tmp/{} to alnair:{}".format(info['Key'], hash_key))
                                value = f.read(max_chunk_size)
                                part += 1
                else:                    
                    s3obj['HashKey'] = saved_keys[key]['HashKey']
                    obj_chunks.append(s3obj)
                    
                    # logger.info('Key {} exists in Alnair.'.format(info['Key']))
                key_lookup[key] = obj_chunks
                return obj_chunks
            
            obj_chunks = []
            max_chunk_size = self.manager.calculate_max_chunk_size()
            with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                futures = []
                for obj in bucket_objs:
                    futures.append(executor.submit(copy_data, obj, max_chunk_size))
                for future in concurrent.futures.as_completed(futures):
                    obj_chunks.extend(future.result())
            
            # create job key_lookup in NFS
            with open("/nfs-master/{}/key_lookup.json".format(jobId), 'w') as f:
                json.dump(key_lookup, f)
            
            # save jobinfo to database
            chunks = []
            now = datetime.utcnow().timestamp()
            for ck in obj_chunks:
                ck['TotalAccessTime'] = 0
                ck['LastAccessTime'] = bson.timestamp.Timestamp(int(now), inc=1)
                chunks.append(ck)
            jobInfo = {
                "meta": {
                    "username": cred.username,
                    "jobId": jobId,
                    "nodeIP": request.nodeIP,
                    "datasource": MessageToDict(request.datasource),
                    "resourceInfo": MessageToDict(request.resource),
                    "createTime": bson.timestamp.Timestamp(int(now), inc=1)
                },
                "QoS": MessageToDict(request.qos),
                "policy": {
                    "createTime": bson.timestamp.Timestamp(int(now), inc=1),
                    "chunks": chunks
                }
            }
            result = self.manager.job_col.insert_one(jobInfo)
            if result.acknowledged:
                logger.info('user {} register job {}'.format(cred.username, jobId))
                       
            return pb.RegisterResponse(rc=pb.RC.REGISTERED, regsucc=pb.RegisterSuccess(
                    jinfo=pb.JobInfo(
                        jobId=jobId,
                        createTime=grpc_ts(now)),
                    policy=pb.Policy(key_lookup=list(key_lookup.keys()))))
        elif rc == pb.RC.DISCONNECTED:
            return pb.RegisterResponse(rc=pb.RC.FAILED, regerr=pb.RegisterError(error="failed to register the jod, user is not connected."))
        else:
            return pb.RegisterResponse(rc=pb.RC.FAILED, regerr=pb.RegisterError(error="failed to register the jod"))

    def deresgister(self, request, context):
        cred = request.cred
        jobId = request.jinfo.jobId
        rc = self.manager.auth_client(cred)
        if rc in [pb.RC.CONNECTED, pb.RC.DISCONNECTED]:
            result = self.manager.job_col.delete_one(filter={"jobId": jobId}) 
            if result.acknowledged and result.deleted_count == 1:
                resp = pb.DeregisterResponse("successfully deregister job {}".format(jobId))
                logger.info('user {} deregister job {}'.format(cred.username, jobId))
            else:
                resp = pb.DeregisterResponse(response='failed to deregister job {}'.format(jobId))
        else:
            resp = pb.DeregisterResponse(response="failed to deregister job {}".format(jobId))
        return resp
      

# TODO: CacheMiss occurs if a key is not available on NFS
class CacheMissService(pb_grpc.CacheMissServicer):
    def __init__(self, manager: Manager) -> None:
        super().__init__()
        self.manager = manager
        
    def call(self, request, context):
        cred = request.cred
        rc = self.manager.auth_client(cred, conn_check=True)
        if rc != pb.RC.CONNECTED:
            return
        chunk = self.manager.job_col.aggregate([
            {"$project": {
                "_id": 0, 
                "chunk": {
                    "$filter": {
                        "input": "$policy.chunks", 
                        "as": "chunks", 
                        "cond": {"$eq": ["$$chunks.Key", request.key]}}
                    }
                }
            },
            {"$unwind": "$chunk"}
        ]).next()['chunk']
        
        return pb.CacheMissResponse(response=True)


if __name__ == '__main__':
    manager = Manager()
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()))
    pb_grpc.add_ConnectionServicer_to_server(ConnectionService(manager), server)
    pb_grpc.add_RegistrationServicer_to_server(RegistrationService(manager), server)
    pb_grpc.add_CacheMissServicer_to_server(CacheMissService(manager), server)
    server.add_insecure_port(address="{}:{}".format(manager.managerconf['bind'], manager.managerconf['port']))
    server.start()
    server.wait_for_termination()