package main

import (
	dps "alnair-device-plugin/pkg/devicepluginserver"
	"log"
	"time"
)

func main() {
	if err := dps.StartDevicePluginServers(); err != nil {
		log.Fatalf("failed to start device plugin server: %v", err)
	}

	// keep main thread running
	for {
		time.Sleep(5 * time.Second)
	}

}
