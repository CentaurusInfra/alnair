package main

import (
	"fmt"
	"math/rand"
	"os"
	"time"

    "github.com/YHDING23/AI-SIG/autonomous-scheduler/UtilSched/pkg/plugins/register"
    "k8s.io/component-base/logs"

)

func main() {
	rand.Seed(time.Now().UTC().UnixNano())

	cmd := register.Register()
	logs.InitLogs()
	defer logs.FlushLogs()
	if err := cmd.Execute(); err != nil {
		_, _ = fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
}