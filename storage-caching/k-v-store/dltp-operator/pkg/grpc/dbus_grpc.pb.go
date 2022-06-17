// Code generated by protoc-gen-go-grpc. DO NOT EDIT.
// versions:
// - protoc-gen-go-grpc v1.2.0
// - protoc             v3.6.1
// source: dbus.proto

package __

import (
	context "context"
	empty "github.com/golang/protobuf/ptypes/empty"
	grpc "google.golang.org/grpc"
	codes "google.golang.org/grpc/codes"
	status "google.golang.org/grpc/status"
)

// This is a compile-time assertion to ensure that this generated file
// is compatible with the grpc package it is being compiled against.
// Requires gRPC-Go v1.32.0 or later.
const _ = grpc.SupportPackageIsVersion7

// ConnectionClient is the client API for Connection service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://pkg.go.dev/google.golang.org/grpc/?tab=doc#ClientConn.NewStream.
type ConnectionClient interface {
	Connect(ctx context.Context, in *ConnectRequest, opts ...grpc.CallOption) (*ConnectResponse, error)
}

type connectionClient struct {
	cc grpc.ClientConnInterface
}

func NewConnectionClient(cc grpc.ClientConnInterface) ConnectionClient {
	return &connectionClient{cc}
}

func (c *connectionClient) Connect(ctx context.Context, in *ConnectRequest, opts ...grpc.CallOption) (*ConnectResponse, error) {
	out := new(ConnectResponse)
	err := c.cc.Invoke(ctx, "/dbus.Connection/connect", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// ConnectionServer is the server API for Connection service.
// All implementations must embed UnimplementedConnectionServer
// for forward compatibility
type ConnectionServer interface {
	Connect(context.Context, *ConnectRequest) (*ConnectResponse, error)
	mustEmbedUnimplementedConnectionServer()
}

// UnimplementedConnectionServer must be embedded to have forward compatible implementations.
type UnimplementedConnectionServer struct {
}

func (UnimplementedConnectionServer) Connect(context.Context, *ConnectRequest) (*ConnectResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method Connect not implemented")
}
func (UnimplementedConnectionServer) mustEmbedUnimplementedConnectionServer() {}

// UnsafeConnectionServer may be embedded to opt out of forward compatibility for this service.
// Use of this interface is not recommended, as added methods to ConnectionServer will
// result in compilation errors.
type UnsafeConnectionServer interface {
	mustEmbedUnimplementedConnectionServer()
}

func RegisterConnectionServer(s grpc.ServiceRegistrar, srv ConnectionServer) {
	s.RegisterService(&Connection_ServiceDesc, srv)
}

func _Connection_Connect_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(ConnectRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(ConnectionServer).Connect(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/dbus.Connection/connect",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(ConnectionServer).Connect(ctx, req.(*ConnectRequest))
	}
	return interceptor(ctx, in, info, handler)
}

// Connection_ServiceDesc is the grpc.ServiceDesc for Connection service.
// It's only intended for direct use with grpc.RegisterService,
// and not to be introspected or modified (even as a copy)
var Connection_ServiceDesc = grpc.ServiceDesc{
	ServiceName: "dbus.Connection",
	HandlerType: (*ConnectionServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "connect",
			Handler:    _Connection_Connect_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "dbus.proto",
}

// RegistrationClient is the client API for Registration service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://pkg.go.dev/google.golang.org/grpc/?tab=doc#ClientConn.NewStream.
type RegistrationClient interface {
	Register(ctx context.Context, in *RegisterRequest, opts ...grpc.CallOption) (*RegisterResponse, error)
	Deregister(ctx context.Context, in *DeregisterRequest, opts ...grpc.CallOption) (*DeregisterResponse, error)
}

type registrationClient struct {
	cc grpc.ClientConnInterface
}

func NewRegistrationClient(cc grpc.ClientConnInterface) RegistrationClient {
	return &registrationClient{cc}
}

func (c *registrationClient) Register(ctx context.Context, in *RegisterRequest, opts ...grpc.CallOption) (*RegisterResponse, error) {
	out := new(RegisterResponse)
	err := c.cc.Invoke(ctx, "/dbus.Registration/register", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *registrationClient) Deregister(ctx context.Context, in *DeregisterRequest, opts ...grpc.CallOption) (*DeregisterResponse, error) {
	out := new(DeregisterResponse)
	err := c.cc.Invoke(ctx, "/dbus.Registration/deregister", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// RegistrationServer is the server API for Registration service.
// All implementations must embed UnimplementedRegistrationServer
// for forward compatibility
type RegistrationServer interface {
	Register(context.Context, *RegisterRequest) (*RegisterResponse, error)
	Deregister(context.Context, *DeregisterRequest) (*DeregisterResponse, error)
	mustEmbedUnimplementedRegistrationServer()
}

// UnimplementedRegistrationServer must be embedded to have forward compatible implementations.
type UnimplementedRegistrationServer struct {
}

func (UnimplementedRegistrationServer) Register(context.Context, *RegisterRequest) (*RegisterResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method Register not implemented")
}
func (UnimplementedRegistrationServer) Deregister(context.Context, *DeregisterRequest) (*DeregisterResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method Deregister not implemented")
}
func (UnimplementedRegistrationServer) mustEmbedUnimplementedRegistrationServer() {}

// UnsafeRegistrationServer may be embedded to opt out of forward compatibility for this service.
// Use of this interface is not recommended, as added methods to RegistrationServer will
// result in compilation errors.
type UnsafeRegistrationServer interface {
	mustEmbedUnimplementedRegistrationServer()
}

func RegisterRegistrationServer(s grpc.ServiceRegistrar, srv RegistrationServer) {
	s.RegisterService(&Registration_ServiceDesc, srv)
}

func _Registration_Register_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(RegisterRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(RegistrationServer).Register(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/dbus.Registration/register",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(RegistrationServer).Register(ctx, req.(*RegisterRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _Registration_Deregister_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(DeregisterRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(RegistrationServer).Deregister(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/dbus.Registration/deregister",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(RegistrationServer).Deregister(ctx, req.(*DeregisterRequest))
	}
	return interceptor(ctx, in, info, handler)
}

// Registration_ServiceDesc is the grpc.ServiceDesc for Registration service.
// It's only intended for direct use with grpc.RegisterService,
// and not to be introspected or modified (even as a copy)
var Registration_ServiceDesc = grpc.ServiceDesc{
	ServiceName: "dbus.Registration",
	HandlerType: (*RegistrationServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "register",
			Handler:    _Registration_Register_Handler,
		},
		{
			MethodName: "deregister",
			Handler:    _Registration_Deregister_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "dbus.proto",
}

// CacheMissClient is the client API for CacheMiss service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://pkg.go.dev/google.golang.org/grpc/?tab=doc#ClientConn.NewStream.
type CacheMissClient interface {
	Call(ctx context.Context, in *CacheMissRequest, opts ...grpc.CallOption) (*CacheMissResponse, error)
}

type cacheMissClient struct {
	cc grpc.ClientConnInterface
}

func NewCacheMissClient(cc grpc.ClientConnInterface) CacheMissClient {
	return &cacheMissClient{cc}
}

func (c *cacheMissClient) Call(ctx context.Context, in *CacheMissRequest, opts ...grpc.CallOption) (*CacheMissResponse, error) {
	out := new(CacheMissResponse)
	err := c.cc.Invoke(ctx, "/dbus.CacheMiss/call", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// CacheMissServer is the server API for CacheMiss service.
// All implementations must embed UnimplementedCacheMissServer
// for forward compatibility
type CacheMissServer interface {
	Call(context.Context, *CacheMissRequest) (*CacheMissResponse, error)
	mustEmbedUnimplementedCacheMissServer()
}

// UnimplementedCacheMissServer must be embedded to have forward compatible implementations.
type UnimplementedCacheMissServer struct {
}

func (UnimplementedCacheMissServer) Call(context.Context, *CacheMissRequest) (*CacheMissResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method Call not implemented")
}
func (UnimplementedCacheMissServer) mustEmbedUnimplementedCacheMissServer() {}

// UnsafeCacheMissServer may be embedded to opt out of forward compatibility for this service.
// Use of this interface is not recommended, as added methods to CacheMissServer will
// result in compilation errors.
type UnsafeCacheMissServer interface {
	mustEmbedUnimplementedCacheMissServer()
}

func RegisterCacheMissServer(s grpc.ServiceRegistrar, srv CacheMissServer) {
	s.RegisterService(&CacheMiss_ServiceDesc, srv)
}

func _CacheMiss_Call_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(CacheMissRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(CacheMissServer).Call(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/dbus.CacheMiss/call",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(CacheMissServer).Call(ctx, req.(*CacheMissRequest))
	}
	return interceptor(ctx, in, info, handler)
}

// CacheMiss_ServiceDesc is the grpc.ServiceDesc for CacheMiss service.
// It's only intended for direct use with grpc.RegisterService,
// and not to be introspected or modified (even as a copy)
var CacheMiss_ServiceDesc = grpc.ServiceDesc{
	ServiceName: "dbus.CacheMiss",
	HandlerType: (*CacheMissServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "call",
			Handler:    _CacheMiss_Call_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "dbus.proto",
}

// HeartbeatClient is the client API for Heartbeat service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://pkg.go.dev/google.golang.org/grpc/?tab=doc#ClientConn.NewStream.
type HeartbeatClient interface {
	Call(ctx context.Context, in *HearbeatMessage, opts ...grpc.CallOption) (*HearbeatMessage, error)
}

type heartbeatClient struct {
	cc grpc.ClientConnInterface
}

func NewHeartbeatClient(cc grpc.ClientConnInterface) HeartbeatClient {
	return &heartbeatClient{cc}
}

func (c *heartbeatClient) Call(ctx context.Context, in *HearbeatMessage, opts ...grpc.CallOption) (*HearbeatMessage, error) {
	out := new(HearbeatMessage)
	err := c.cc.Invoke(ctx, "/dbus.Heartbeat/call", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// HeartbeatServer is the server API for Heartbeat service.
// All implementations must embed UnimplementedHeartbeatServer
// for forward compatibility
type HeartbeatServer interface {
	Call(context.Context, *HearbeatMessage) (*HearbeatMessage, error)
	mustEmbedUnimplementedHeartbeatServer()
}

// UnimplementedHeartbeatServer must be embedded to have forward compatible implementations.
type UnimplementedHeartbeatServer struct {
}

func (UnimplementedHeartbeatServer) Call(context.Context, *HearbeatMessage) (*HearbeatMessage, error) {
	return nil, status.Errorf(codes.Unimplemented, "method Call not implemented")
}
func (UnimplementedHeartbeatServer) mustEmbedUnimplementedHeartbeatServer() {}

// UnsafeHeartbeatServer may be embedded to opt out of forward compatibility for this service.
// Use of this interface is not recommended, as added methods to HeartbeatServer will
// result in compilation errors.
type UnsafeHeartbeatServer interface {
	mustEmbedUnimplementedHeartbeatServer()
}

func RegisterHeartbeatServer(s grpc.ServiceRegistrar, srv HeartbeatServer) {
	s.RegisterService(&Heartbeat_ServiceDesc, srv)
}

func _Heartbeat_Call_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(HearbeatMessage)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(HeartbeatServer).Call(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/dbus.Heartbeat/call",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(HeartbeatServer).Call(ctx, req.(*HearbeatMessage))
	}
	return interceptor(ctx, in, info, handler)
}

// Heartbeat_ServiceDesc is the grpc.ServiceDesc for Heartbeat service.
// It's only intended for direct use with grpc.RegisterService,
// and not to be introspected or modified (even as a copy)
var Heartbeat_ServiceDesc = grpc.ServiceDesc{
	ServiceName: "dbus.Heartbeat",
	HandlerType: (*HeartbeatServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "call",
			Handler:    _Heartbeat_Call_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "dbus.proto",
}

// LoggerClient is the client API for Logger service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://pkg.go.dev/google.golang.org/grpc/?tab=doc#ClientConn.NewStream.
type LoggerClient interface {
	Call(ctx context.Context, opts ...grpc.CallOption) (Logger_CallClient, error)
}

type loggerClient struct {
	cc grpc.ClientConnInterface
}

func NewLoggerClient(cc grpc.ClientConnInterface) LoggerClient {
	return &loggerClient{cc}
}

func (c *loggerClient) Call(ctx context.Context, opts ...grpc.CallOption) (Logger_CallClient, error) {
	stream, err := c.cc.NewStream(ctx, &Logger_ServiceDesc.Streams[0], "/dbus.Logger/call", opts...)
	if err != nil {
		return nil, err
	}
	x := &loggerCallClient{stream}
	return x, nil
}

type Logger_CallClient interface {
	Send(*LogItem) error
	CloseAndRecv() (*empty.Empty, error)
	grpc.ClientStream
}

type loggerCallClient struct {
	grpc.ClientStream
}

func (x *loggerCallClient) Send(m *LogItem) error {
	return x.ClientStream.SendMsg(m)
}

func (x *loggerCallClient) CloseAndRecv() (*empty.Empty, error) {
	if err := x.ClientStream.CloseSend(); err != nil {
		return nil, err
	}
	m := new(empty.Empty)
	if err := x.ClientStream.RecvMsg(m); err != nil {
		return nil, err
	}
	return m, nil
}

// LoggerServer is the server API for Logger service.
// All implementations must embed UnimplementedLoggerServer
// for forward compatibility
type LoggerServer interface {
	Call(Logger_CallServer) error
	mustEmbedUnimplementedLoggerServer()
}

// UnimplementedLoggerServer must be embedded to have forward compatible implementations.
type UnimplementedLoggerServer struct {
}

func (UnimplementedLoggerServer) Call(Logger_CallServer) error {
	return status.Errorf(codes.Unimplemented, "method Call not implemented")
}
func (UnimplementedLoggerServer) mustEmbedUnimplementedLoggerServer() {}

// UnsafeLoggerServer may be embedded to opt out of forward compatibility for this service.
// Use of this interface is not recommended, as added methods to LoggerServer will
// result in compilation errors.
type UnsafeLoggerServer interface {
	mustEmbedUnimplementedLoggerServer()
}

func RegisterLoggerServer(s grpc.ServiceRegistrar, srv LoggerServer) {
	s.RegisterService(&Logger_ServiceDesc, srv)
}

func _Logger_Call_Handler(srv interface{}, stream grpc.ServerStream) error {
	return srv.(LoggerServer).Call(&loggerCallServer{stream})
}

type Logger_CallServer interface {
	SendAndClose(*empty.Empty) error
	Recv() (*LogItem, error)
	grpc.ServerStream
}

type loggerCallServer struct {
	grpc.ServerStream
}

func (x *loggerCallServer) SendAndClose(m *empty.Empty) error {
	return x.ServerStream.SendMsg(m)
}

func (x *loggerCallServer) Recv() (*LogItem, error) {
	m := new(LogItem)
	if err := x.ServerStream.RecvMsg(m); err != nil {
		return nil, err
	}
	return m, nil
}

// Logger_ServiceDesc is the grpc.ServiceDesc for Logger service.
// It's only intended for direct use with grpc.RegisterService,
// and not to be introspected or modified (even as a copy)
var Logger_ServiceDesc = grpc.ServiceDesc{
	ServiceName: "dbus.Logger",
	HandlerType: (*LoggerServer)(nil),
	Methods:     []grpc.MethodDesc{},
	Streams: []grpc.StreamDesc{
		{
			StreamName:    "call",
			Handler:       _Logger_Call_Handler,
			ClientStreams: true,
		},
	},
	Metadata: "dbus.proto",
}
