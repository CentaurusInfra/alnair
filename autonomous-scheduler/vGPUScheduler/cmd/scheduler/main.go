package main

import (
	"fmt"
	"math/rand"
	"os"
	"time"

	"vGPUScheduler/pkg/vGPUScheduler"
	"k8s.io/component-base/logs"
	"k8s.io/kubernetes/cmd/kube-scheduler/app"

	_ "sigs.k8s.io/scheduler-plugins/pkg/apis/config/scheme"
)

func main() {
	rand.Seed(time.Now().UTC().UnixNano())
	logs.InitLogs()
	defer logs.FlushLogs()

	cmd := app.NewSchedulerCommand(
		app.WithPlugin(vGPUScheduler.Name, vGPUScheduler.New),
	)

	if err := cmd.Execute(); err != nil {
		_, _ = fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
}
