package cgroupserver

import (
	"bufio"
	"context"
	"log"
	"net"
	"os"
	"path"
	"strings"

	dockerclient "github.com/docker/docker/client"
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
	log.Printf("vGPU server starts, listening to unix socket: %v", AlnairCgroupServerSocket)
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

// implements registercgroup in another way, contact docker socket, with containerTop to get all the pids
// based on the registed cgroup process, write PIDs and ContainerID two files in container's Alnair workspace
func registerCgroup(cgroup, alnairID string) error {
	log.Printf("Received registration request for cgroup: %v", cgroup)
	//get container ID and all pids in the container through dockerclient containerTop function
	_, containerId := parsePodIDContainerID(cgroup)
	pids := containerTopPids(containerId)

	//write pids to cgroup.procs file
	containerWorkspace := path.Join(AlnairContainerWorkspaceRoot, alnairID)
	procsFile := path.Join(containerWorkspace, "cgroup.procs")
	os.RemoveAll(procsFile)
	f, err := os.Create(procsFile)
	if err != nil {
		log.Printf("cannot create file: %s", procsFile)
		return err
	}
	for _, pid := range pids {
		f.WriteString(pid + "\n")
	}
	f.Close()
	log.Printf("cgroup.procs file is ready with pids:%v", pids)

	//Write containerId to containerID file
	containerIdFilepath := path.Join(containerWorkspace, "containerID")
	if err := os.WriteFile(containerIdFilepath, []byte(containerId), 0644); err != nil {
		log.Printf("ERROR: failed to write container ID %v for alnair ID %v", containerId, alnairID)
		return err
	}
	log.Printf("containerID file is ready with ID:%v", containerId)
	return nil
}

func containerTopPids(containerID string) (pids []string) {
	cli, err := dockerclient.NewClientWithOpts(dockerclient.FromEnv)
	if err != nil {
		panic(err)
	}
	var args []string
	topBody, err := cli.ContainerTop(context.Background(), containerID, args)
	if err != nil {
		panic(err)
	}
	for _, process := range topBody.Processes {
		pids = append(pids, process[1]) //topBody.Titles is [UID PID PPID C STIME TTY TIME CMD]
	}
	return
}

//limits: besteffort pod, should support burstable as well
//example cgroup path kubernetes v 1.21.X:
//kubepods/besteffort/podb494d806-bfe7-4c33-8e23-032da1434a90/06b159b3f1cb4c021766a97e5ac82d18284c381223e5539aa510269ee5eed4d3
//example cgroup path kubernetes v 1.20.X:
//kubepods.slice/kubepods-besteffort.slice/kubepods-besteffort-pod227905f8_727f_4a84_9883_25880640c810.slice/docker-70b480ffcc00feec75639fdecdf105ebcbb115fa30f78ae16b380f450d8dc7b0.scope
func parsePodIDContainerID(cgroup string) (podId, containerId string) {
	a := strings.Split(cgroup, "/")
	for _, str := range a {
		if strings.HasPrefix(str, "kubepods-besteffort-pod") { //k8s v1.20
			r := strings.NewReplacer("kubepods-besteffort-pod", "", ".slice", "", "_", "-")
			podId = r.Replace(str)
		}
		if strings.HasPrefix(str, "pod") { //k8s v1.21
			r := strings.NewReplacer("pod", "")
			podId = r.Replace(str)
		}
		if strings.HasPrefix(str, "docker-") { //k8s v1.20
			r := strings.NewReplacer("docker-", "", ".scope", "")
			containerId = r.Replace(str)
		}
	}
	if len(containerId) == 0 { //k8s v1.21, use the last element of the cgroup after split by
		containerId = a[len(a)-1]
	}

	if len(podId) == 0 {
		log.Printf("podID extraction failed!")
	}
	if len(containerId) == 0 {
		log.Printf("containerID extraction failed!")
	}
	return
}
