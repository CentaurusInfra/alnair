#!/bin/bash

# 命令行参数的说明：
# 1）--proto_path：指定 .proto 查找路径
# 2）--python_out：databus_pb2.py 的输出路径
# 3）--grpc_python_out：databus_pb2_grpc.py 的输出路径
# databus.proto：协议文件

python3 -m grpc_tools.protoc --proto_path=./ --python_out=./ --grpc_python_out=./ ./dbus.proto