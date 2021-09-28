package plugins

import (
	"github.com/YHDING23/UtilSched/pkg/plugins/utilsched"
	"github.com/spf13/cobra"
	"k8s.io/kubernetes/cmd/kube-scheduler/app"
)

func Register() *cobra.Command {
	return app.NewSchedulerCommand(
		app.WithPlugin(utilsched.Name, utilsched.New),
	)
}