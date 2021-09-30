package utilsched

import (
	"context"
	"errors"
	"fmt"
	"math"

	"UtilSched/pkg/plugins/utilsched/collection"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/klog"
	framework "k8s.io/kubernetes/pkg/scheduler/framework"
	ctrl "sigs.k8s.io/controller-runtime"
	"strconv"
)

const (
	// Name is plugin name
	Name = "utilsched"
)

var (
	_ framework.ScorePlugin     = &UtilSched{}
	_ framework.ScoreExtensions = &UtilSched{}

//     scheme = runtime.NewScheme()
)

const (
	CPUUsageWeight        = 3
	GPUModelWeight_k80    = 6
	GPUModelWeight_TITANX = 7
	GPUModelWeight_V100   = 10
	ActualWeight          = 10
	NodeWeight            = 5
)

type UtilSched struct {
	// 	args   *Args
	handle framework.Handle
}

// type Args struct {
// 	KubeConfig string `json:"kubeconfig,omitempty"`
// 	Master     string `json:"master,omitempty"`
// }

func New(_ runtime.Object, handle framework.Handle) (framework.Plugin, error) {
	mgrConfig := ctrl.GetConfigOrDie()
	mgrConfig.QPS = 1000
	mgrConfig.Burst = 1000
	mgr, err := ctrl.NewManager(mgrConfig, ctrl.Options{
		MetricsBindAddress: "",
		LeaderElection:     false,
		Port:               9443,
	})
	go func() {
		if err = mgr.Start(ctrl.SetupSignalHandler()); err != nil {
			klog.Error(err)
			panic(err)
		}
	}()

	if err != nil {
		klog.Error(err)
		return nil, err
	}
	return &UtilSched{
		// 		args:   args,
		handle: handle,
	}, nil
}

func (u *UtilSched) Name() string {
	return Name
}

func (u *UtilSched) Score(ctx context.Context, state *framework.CycleState, p *v1.Pod, nodeName string) (int64, *framework.Status) {
	// Get Node Info
	nodeInfo, err := u.handle.SnapshotSharedLister().NodeInfos().Get(nodeName)
	if err != nil {
		return 0, framework.NewStatus(framework.Error, fmt.Sprintf("getting node %q from Snapshot: %v", nodeName, err))
	}
	_ = collection.CollectValues(state, nodeName)

	uNodeScore, err := CalculateScore(state, p, nodeName)
	if err != nil {
		klog.Errorf("CalculateScore Error: %v", err)
		return 0, framework.NewStatus(framework.Error, fmt.Sprintf("Score Node: %v Error: %v", nodeInfo.Node().Name, err))
	}
	nodeScore := Uint64ToInt64(uNodeScore)
	return nodeScore, framework.NewStatus(framework.Success, "")
}

func CalculateScore(state *framework.CycleState, pod *v1.Pod, nodeName string) (uint64, error) {
	state.RLock()
	d, err := state.Read("Anno")
	state.RUnlock()
	if err != nil {
		return 0, errors.New("Error Get CycleState Info Annotation Error: " + err.Error())
	}
	///////
	data, ok := d.(*collection.Data)
	//////
	if !ok {
		return 0, errors.New("The Type is not Data ")
	}
	return CalculateNodeScore(data.NodeValue)*NodeWeight + CalculateActualScore(data.GPUValue)*ActualWeight + CurPodCPUUsageScore(pod)*CPUUsageWeight, nil
}

func CalculateNodeScore(nodeValue collection.Node_static) uint64 {
	modelScore := uint64(0)
	switch nodeValue.Model {
	case "Tesla K80":
		modelScore = uint64(nodeValue.MemSize * uint64(GPUModelWeight_k80) * uint64(nodeValue.Count) / 12)
	case "TITAN X (Pascal)":
		modelScore = uint64(nodeValue.MemSize * uint64(GPUModelWeight_TITANX) * uint64(nodeValue.Count) / 12)
	case "V100":
		modelScore = uint64(nodeValue.MemSize * uint64(GPUModelWeight_V100) * uint64(nodeValue.Count) / 12)
	}
	return modelScore
}

func CalculateActualScore(gpuvalue map[string]collection.GPU_mem_usage) uint64 {
	actualScore := uint64(0)
	for _, card := range gpuvalue {
		actualScore += uint64(card.Mem_used / (card.Mem_used + card.Mem_free))
	}
	return actualScore
}

func CurPodCPUUsageScore(pod *v1.Pod) uint64 {
	var curPodCPUUsage int64
	for _, container := range pod.Spec.Containers {
		curPodCPUUsage += PredictUtilisation(&container)
	}
	return uint64(curPodCPUUsage)
}

// Predict utilization for a container based on its requests/limits
func PredictUtilisation(container *v1.Container) int64 {
	if _, ok := container.Resources.Limits[v1.ResourceCPU]; ok {
		return container.Resources.Limits.Cpu().MilliValue()
	} else if _, ok := container.Resources.Requests[v1.ResourceCPU]; ok {
		return int64(math.Round(float64(container.Resources.Requests.Cpu().MilliValue())))
	} else {
		// 		return requestsMilliCores
		return int64(1)
	}
}

func (u *UtilSched) NormalizeScore(_ context.Context, _ *framework.CycleState, pod *v1.Pod, scores framework.NodeScoreList) *framework.Status {
	var (
		highest int64 = 0
		lowest        = scores[0].Score
	)

	for _, nodeScore := range scores {
		if nodeScore.Score < lowest {
			lowest = nodeScore.Score
		}
		if nodeScore.Score > highest {
			highest = nodeScore.Score
		}
	}

	if highest == lowest {
		lowest--
	}

	// Set Range to [0-100]
	for i, nodeScore := range scores {
		scores[i].Score = (nodeScore.Score - lowest) * framework.MaxNodeScore / (highest - lowest)
		klog.Infof("Node: %v, Score: %v in Plugin: UtilSched When scheduling Pod: %v/%v", scores[i].Name, scores[i].Score, pod.GetNamespace(), pod.GetName())
	}
	return nil
}

func (u *UtilSched) ScoreExtensions() framework.ScoreExtensions {
	return u
}

func Uint64ToInt64(intNum uint64) int64 {
	return StrToInt64(strconv.FormatUint(intNum, 10))
}

func StrToInt64(str string) int64 {
	if i, e := strconv.Atoi(str); e != nil {
		return 0
	} else {
		return int64(i)
	}
}
