package controllers

import (
	"encoding/json"
)

type QoS struct {
	UseCache         bool
	MaxMemory        int
	DurabilityInDisk int
}

var DefaultQoS = QoS{
	UseCache:         true,
	MaxMemory:        0,
	DurabilityInDisk: 1440,
}

// convert QoS configuration to K8s ConfigMap
func ToConfigmapData(q QoS, binary bool) interface{} {
	data, _ := json.Marshal(q)
	if binary {
		var result map[string][]byte
		json.Unmarshal(data, &result)
		return result
	} else {
		var result map[string]string
		json.Unmarshal(data, &result)
		return result
	}
}
