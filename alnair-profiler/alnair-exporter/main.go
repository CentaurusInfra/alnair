package main

import (
	"alnair-profiler/pkg/exporter"
	"log"
	"os"
	"strings"
)

func main() {
	//get nodeName from environment variable, which is passed during pod creation
	envs := os.Environ()
	nodeName := ""
	podIP := ""
	for _, env := range envs {
		variable := strings.Split(env, "=")
		if variable[0] == "MY_NODE_NAME" {
			nodeName = variable[1]
			log.Printf("My node name is %s", nodeName)
		}
		if variable[0] == "MY_POD_IP" {
			podIP = variable[1]
			log.Printf("My pod IP is %s", podIP)
		}
	}
	// set an port to lanuch exporter
	port := ":9876"
	exporter.Start(nodeName, podIP, port)
}
