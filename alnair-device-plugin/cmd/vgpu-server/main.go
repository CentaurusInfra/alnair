package main

import (
	vs "alnair-device-plugin/pkg/vgpuserver"
	"time"
)

func main() {
	vGPUServer := vs.NewVGPUServer()
	go vGPUServer.Start()

	// keep main thread running
	for {
		time.Sleep(5 * time.Second)
	}
}
