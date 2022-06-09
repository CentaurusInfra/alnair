from __future__ import print_function

import grpc

import utils.databus.databus_pb2
import utils.databus.databus_pb2_grpc
from google.protobuf.json_format import MessageToJson


"""
optional int32 logFreq = 5; // frequency of collecting logs
optional int32 logLinger = 6; // batch log every `logLinger` seconds
optional int32 logBatchSize = 7; // batch log every `logBatchSize` items
"""