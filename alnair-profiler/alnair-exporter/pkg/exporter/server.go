package exporter

import (
	"log"
	"net/http"

	"alnair-profiler/pkg/nvmlcollect"
	"alnair-profiler/pkg/cpucollect"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

//start a http server and respond to /metrics<port> end point
func Start(nodeName string, port string) {
	//remove two default metrics collector
	prometheus.Unregister(prometheus.NewProcessCollector(prometheus.ProcessCollectorOpts{}))
	prometheus.Unregister(prometheus.NewGoCollector())

	//register nvml collector, metrics from nvml server, pass nodeName to metrics
	nvml := nvmlcollect.NewCollector(nodeName)
	prometheus.MustRegister(nvml)

	//register cpu stats collector, metrics from /proc/stat file
	cpu := cpucollect.NewCollector(nodeName)
	prometheus.MustRegister(cpu)

	//Start the HTTP server and expose metrics at /metrics
	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.Handler())
	log.Printf("Alnair exporter beginning to serve on port %s", port)
	err := http.ListenAndServe(port, mux)
	if err != nil {
		log.Fatalf("cannot start exporter: %s", err)
	}

}
