package main

import (
	"fmt"
	"math/rand"
	"os"
	"time"

	"k8s.io/component-base/logs"
	"k8s.io/kubernetes/cmd/kube-scheduler/app"
	// 	"vGPUScheduler/pkg/vGPUScheduler"
	"vGPUScheduler/pkg/alnair-cost-saving"
	"vGPUScheduler/pkg/alnair-high-performance"

	_ "sigs.k8s.io/scheduler-plugins/pkg/apis/config/scheme"
)

func main() {
	rand.Seed(time.Now().UTC().UnixNano())
	logs.InitLogs()
	defer logs.FlushLogs()

	cmd := app.NewSchedulerCommand(
		// 	    app.WithPlugin(vGPUScheduler.Name, vGPUScheduler.New),
		app.WithPlugin(alnairhighperformance.Name, alnairhighperformance.New),
		app.WithPlugin(alnaircostsaving.Name, alnaircostsaving.New),
	)

	if err := cmd.Execute(); err != nil {
		_, _ = fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
}
