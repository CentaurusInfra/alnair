package main

import (
	ds "alnair-device-plugin/pkg/devicepluginserver"
	"log"
	"time"
)

func main() {
	devicePluginServer := ds.NewDevicePluginServer()
	if err := devicePluginServer.Start(); err != nil {
		log.Fatalf("failed to start device plugin server: %v", err)
	}

	// keep main thread running
	for {
		time.Sleep(5 * time.Second)
	}

}
