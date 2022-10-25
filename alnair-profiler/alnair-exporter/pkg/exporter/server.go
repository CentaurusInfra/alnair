package exporter

import (
	"fmt"
	"log"
	"net/http"

	"alnair-profiler/pkg/cpucollect"
	"alnair-profiler/pkg/cudacollect"
	"alnair-profiler/pkg/nvmlcollect"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

//start a http server and respond to /metrics<port> end point
func Start(nodeName string, podIP string, port string) {
	//remove two default metrics collector
	prometheus.Unregister(prometheus.NewProcessCollector(prometheus.ProcessCollectorOpts{}))
	prometheus.Unregister(prometheus.NewGoCollector())

	//register nvml collector, metrics from nvml server, pass nodeName to metrics
	nvml := nvmlcollect.NewCollector(nodeName)
	prometheus.MustRegister(nvml)

	//register cpu stats collector, metrics from /proc/stat file
	cpu := cpucollect.NewCollector(nodeName)
	prometheus.MustRegister(cpu)

	//register cuda stats collector, metrics from /var/lib/alnair/workspace
	cuda := cudacollect.NewCollector(nodeName)
	prometheus.MustRegister(cuda)

	//Start the HTTP server and expose metrics at /metrics
	mux := http.NewServeMux()
	mux.Handle("/", http.HandlerFunc(mainPage))
	mux.Handle("/metrics", promhttp.Handler())
	log.Printf("Alnair exporter beginning to serve on port %s", port)
	log.Printf("see available metrics with: wget http://%s%s/metrics", podIP, port)
	err := http.ListenAndServe(port, mux)
	if err != nil {
		log.Fatalf("cannot start exporter: %s", err)
	}

}

func mainPage(w http.ResponseWriter, req *http.Request) {
	fmt.Fprintf(w, "hello, this is alnair exporter, please scrape /metrics endpoints, default port 9876\n")
}
