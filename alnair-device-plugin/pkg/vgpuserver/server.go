package cgroupserver

import (
	"bufio"
	"io/ioutil"
	"log"
	"net"
	"os"
	"path"
	"strings"
)

const (
	AlnairCgroupServerSocket     = "/run/alnair/alnair.sock"
	AlnairContainerWorkspaceRoot = "/var/lib/alnair/workspace"
)

// VGPUServer listens to requests from containers, sets up a vGPU workspace for each container
// TODO: add support for cgroup driver cgroupfs
type VGPUServer struct {
	stop chan interface{}
}

func NewVGPUServer() *VGPUServer {
	return &VGPUServer{
		stop: make(chan interface{}),
	}
}

func (cs *VGPUServer) Start() {
	if err := os.RemoveAll(AlnairCgroupServerSocket); err != nil {
		log.Fatal(err)
	}

	l, err := net.Listen("unix", AlnairCgroupServerSocket)
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}
        log.Printf("Listen to unix socket: %v", AlnairCgroupServerSocket)
	defer l.Close()

	for {
		c, err := l.Accept()
		if err != nil {
			log.Fatal(err)
		}

		go handleConnection(c)
	}
}

func handleConnection(c net.Conn) {
	defer c.Close()
	input, err := bufio.NewReader(c).ReadString('\n')
	if err != nil {
		log.Fatal(err)
	}
	input = input[:len(input)-1]
	items := strings.Split(input, " ")
	err = registerCgroup(items[0], items[1])
	if err != nil {
		c.Write([]byte(err.Error()))
	} else {
		c.Write([]byte("ok"))
	}
}

// Register implements registration.Register
func registerCgroup(cgroup, alnairID string) error {
	log.Printf("Received registration request for cgroup: %v", cgroup)
	pidsfile := path.Join("/sys/fs/cgroup/memory", cgroup, "cgroup.procs")

	if _, err := os.Stat(pidsfile); os.IsNotExist(err) {
		log.Printf("cannot find cgroup.procs file %v: %v", pidsfile, err)
		return err
	}

	containerWorkspace := path.Join(AlnairContainerWorkspaceRoot, alnairID)
	copyto := path.Join(containerWorkspace, "cgroup.procs")
	os.RemoveAll(copyto)

	if err := copyfile(pidsfile, copyto); err != nil {
		log.Printf("cannot copy from %v to %v", pidsfile, copyto)
		return err
	}

	_, containerId := parsePodIDContainerID(cgroup)
	containerIdFilepath := path.Join(containerWorkspace, "containerID")

	if err := os.WriteFile(containerIdFilepath, []byte(containerId), 0644); err != nil {
		log.Printf("ERROR: failed to write container ID %v for alnair ID %v", containerId, alnairID)
		return err
	}

	return nil
}

func copyfile(src, dst string) error {
	in, err := ioutil.ReadFile(src)
	if err != nil {
		return err
	}
	err = ioutil.WriteFile(dst, in, 0644)
	if err != nil {
		return err
	}
	return nil
}

func parsePodIDContainerID(cgroup string) (podId, containerId string) {
	a := strings.Split(cgroup, "/")
	for _, str := range a {
		if strings.HasPrefix(str, "kubepods-besteffort-pod") {
			r := strings.NewReplacer("kubepods-besteffort-pod", "", ".slice", "", "_", "-")
			podId = r.Replace(str)
		}
		if strings.HasPrefix(str, "docker-") {
			r := strings.NewReplacer("docker-", "", ".scope", "")
			containerId = r.Replace(str)
		}
	}
	return
}
