package devicepluginserver

import (
	"context"
	"fmt"
	"log"
	"net"
	"os"
	"path"
	"sort"
	"strings"
	"time"

	"github.com/NVIDIA/gpu-monitoring-tools/bindings/go/nvml"
	"google.golang.org/grpc"
	utilrand "k8s.io/apimachinery/pkg/util/rand"
	pluginapi "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
)

const (
	devicePluginServerSocket = "alnair-gpu.sock"
	devicePluginResourceName = "alnair/vgpu-mem"
	gpuMemoryChunkSize       = 1073741824 // GiB
	alnairInterposeLibPath   = "/opt/alnair/libcuinterpose.so"
)

type DevicePluginServer struct {
	pluginapi.UnimplementedDevicePluginServer
	server *grpc.Server
	stop   chan interface{}
}

func NewDevicePluginServer() *DevicePluginServer {
	return &DevicePluginServer{
		server: grpc.NewServer(),
		stop:   make(chan interface{}),
	}
}

func (s *DevicePluginServer) Start() error {
	if err := nvml.Init(); err != nil {
		return err
	}

	// Start grpc server
	sock := path.Join(pluginapi.DevicePluginPath, devicePluginServerSocket)

	if err := os.RemoveAll(sock); err != nil && err != os.ErrNotExist {
		return err
	}

	l, err := net.Listen("unix", sock)
	if err != nil {
		return err
	}

	pluginapi.RegisterDevicePluginServer(s.server, s)
	go func() {
		if err := s.server.Serve(l); err != nil {
			log.Fatalf("failed to serve grpc: %v", err)
		}
	}()

	conn, err := dialGrpc(sock)
	if err != nil {
		log.Println("failed to wait for grpc server to be ready")
		return err
	}
	conn.Close()

	// register with kubelet
	if err := s.RegisterWithKubelet(); err != nil {
		log.Println("failed to register with kubelet")
		return err
	}

	return nil
}

func (s *DevicePluginServer) Stop() {
	close(s.stop)
	s.server.Stop()
}

func (s *DevicePluginServer) RegisterWithKubelet() error {
	conn, err := dialGrpc(pluginapi.KubeletSocket)
	if err != nil {
		log.Println("failed to dail kubelet grpc endpoint")
		return err
	}

	client := pluginapi.NewRegistrationClient(conn)
	request := &pluginapi.RegisterRequest{
		Version:      pluginapi.Version,
		Endpoint:     devicePluginServerSocket,
		ResourceName: devicePluginResourceName,
		Options: &pluginapi.DevicePluginOptions{
			GetPreferredAllocationAvailable: true,
		},
	}

	if _, err = client.Register(context.Background(), request); err != nil {
		return err
	}

	return nil
}

func (s *DevicePluginServer) ListAndWatch(e *pluginapi.Empty, lws pluginapi.DevicePlugin_ListAndWatchServer) error {
	devs := getDevices()
	lws.Send(&pluginapi.ListAndWatchResponse{Devices: devs})
	<-s.stop
	return nil
}

func (s *DevicePluginServer) Allocate(ctx context.Context, req *pluginapi.AllocateRequest) (*pluginapi.AllocateResponse, error) {
	var resp pluginapi.AllocateResponse
	for _, creq := range req.ContainerRequests {
		devIDs := getRealDeviceIDs(creq.DevicesIDs)

		alnairID := utilrand.String(5)
		hostWorkspacePath := path.Join("/var/lib/alnair/workspace", alnairID)
		if err := os.MkdirAll(hostWorkspacePath, 0700); err != nil {
			log.Printf("ERROR: failed to create alnair workspace %s: %v", hostWorkspacePath, err)
		}
		limitsFilepath := path.Join(hostWorkspacePath, "limits")
		limits := fmt.Sprintf("vmem:%d", len(creq.DevicesIDs))
		if err := os.WriteFile(limitsFilepath, []byte(limits), 0644); err != nil {
			log.Printf("ERROR: failed to write alnair resource limits")
		}

		var cresp pluginapi.ContainerAllocateResponse
		cresp.Envs = map[string]string{
			"NVIDIA_VISIBLE_DEVICES": strings.Join(devIDs, ","),
			"ALNAIR_ID":              alnairID,
			"ALNAIR_WORKSPACE_PATH":  "/var/lib/alnair/workspace",
			"ALNAIR_SOCKET":          "/run/alnair.sock",
			"LD_PRELOAD":             alnairInterposeLibPath,
		}
		cresp.Mounts = []*pluginapi.Mount{
			{
				ContainerPath: "/var/lib/alnair/workspace",
				HostPath:      hostWorkspacePath,
			},
		}
		resp.ContainerResponses = append(resp.ContainerResponses, &cresp)
	}
	return &resp, nil
}

func (s *DevicePluginServer) GetPreferredAllocation(ctx context.Context, req *pluginapi.PreferredAllocationRequest) (*pluginapi.PreferredAllocationResponse, error) {
	var ret pluginapi.PreferredAllocationResponse
	for _, creq := range req.ContainerRequests {
		preferredDeviceIDs := getPreferredDeviceIDs(creq.AvailableDeviceIDs, creq.AllocationSize)
		ret.ContainerResponses = append(ret.ContainerResponses,
			&pluginapi.ContainerPreferredAllocationResponse{
				DeviceIDs: preferredDeviceIDs,
			},
		)
	}

	return &ret, nil
}

func (s *DevicePluginServer) GetDevicePluginOptions(ctx context.Context, e *pluginapi.Empty) (*pluginapi.DevicePluginOptions, error) {
	return &pluginapi.DevicePluginOptions{
		GetPreferredAllocationAvailable: true,
	}, nil
}

func getRealDeviceIDs(syntheticIDs []string) []string {
	var ret []string
	sort.Strings(syntheticIDs)
	for _, sid := range syntheticIDs {
		id := strings.SplitN(sid, "_", 2)[0]
		if len(ret) == 0 || id != ret[len(ret)-1] {
			ret = append(ret, id)
		}
	}
	return ret
}

func getPreferredDeviceIDs(availableDeviceIDs []string, allocationSize int32) []string {
	return availableDeviceIDs[0:allocationSize]
}

func getDevices() []*pluginapi.Device {
	n, err := nvml.GetDeviceCount()
	if err != nil {
		panic(err)
	}

	var devs []*pluginapi.Device
	for i := uint(0); i < n; i++ {
		d, err := nvml.NewDevice(i)
		if err != nil {
			panic(err)
		}

		devs = append(devs, getPluginApiDevice(d)...)
	}

	return devs
}

func getPluginApiDevice(d *nvml.Device) []*pluginapi.Device {
	var ret []*pluginapi.Device
	chunkSzInMiB := gpuMemoryChunkSize / 1024 / 1024
	numChunks := (int(*d.Memory) + chunkSzInMiB/2) / chunkSzInMiB
	for i := uint(0); i < uint(numChunks); i++ {
		var t pluginapi.Device
		t.ID = fmt.Sprintf("%s_%d", d.UUID, i)
		t.Health = pluginapi.Healthy
		if d.CPUAffinity != nil {
			t.Topology = &pluginapi.TopologyInfo{
				Nodes: []*pluginapi.NUMANode{
					{ID: int64(*d.CPUAffinity)},
				},
			}
		}
		ret = append(ret, &t)
	}
	return ret
}

func dialGrpc(sock string) (*grpc.ClientConn, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	conn, err := grpc.DialContext(
		ctx,
		"unix://"+sock,
		grpc.WithInsecure(),
		grpc.WithBlock(),
	)

	if err != nil {
		return nil, err
	}

	return conn, nil
}
