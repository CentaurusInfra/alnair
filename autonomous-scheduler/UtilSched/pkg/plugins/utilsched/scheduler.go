package utilsched

import (
	"context"
	"errors"
	"fmt"
	"os"
	"regexp"
	"strconv"

	meta "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog"
	framework "k8s.io/kubernetes/pkg/scheduler/framework"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"

	"github.com/YHDING23/UtilSched/pkg/plugins/utilsched/collection"
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
	args   *Args
	handle framework.Handle
}

type Args struct {
	KubeConfig string `json:"kubeconfig,omitempty"`
	Master     string `json:"master,omitempty"`
}

func New(plArgs *runtime.Unknown, h framework.Handle) (framework.Plugin, error) {
	args := &Args{}
	if err := framework.DecodeInto(plArgs, args); err != nil {
		return nil, err
	}
	klog.V(3).Infof("--------> args: %+v", args)
	return &UtilSched{
		args:   args,
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
	return CalculateNodeScore(data.NodeValue)*NodeWeight + CalculateActualScore(data)*ActualWeight + CurPodCPUUsageScore(pod)*CPUUsageWeight
}

func CalculateBasicScore(nodeValue collection.NodeValue) uint64 {
	switch nodeValue.Model {
	case "Tesla K80":
		modelScore = uint64(nodeValue.MemSize * GPUModelWeight_k80 * nodeValue.Count / 12)
	case "TITAN X (Pascal)":
		modelScore = uint64(nodeValue.MemSize * GPUModelWeight_TITANX * nodeValue.Count / 12)
	case "V100":
		modelScore = uint64(nodeValue.MemSize * GPUModelWeight_V100 * nodeValue.Count / 12)
	}
	return modelScore
}

func CalculateActualScore(data collection.Data) uint64 {
	actualScore := uint64(0)
	for _, card := range data.GPUValue {
		actualScore += uint64(card.mem_used / (card.mem_used + card.mem_free))
	}
	return actualScore
}

func CurPodCPUUsageScore(pod *v1.Pod) uint64 {
	var curPodCPUUsage int64
	for _, container := range pod.Spec.Containers {
		curPodCPUUsage += PredictUtilisation(&container)
	}
}

// Predict utilization for a container based on its requests/limits
func PredictUtilisation(container *v1.Container) int64 {
	if _, ok := container.Resources.Limits[v1.ResourceCPU]; ok {
		return container.Resources.Limits.Cpu().MilliValue()
	} else if _, ok := container.Resources.Requests[v1.ResourceCPU]; ok {
		return int64(math.Round(float64(container.Resources.Requests.Cpu().MilliValue())))
	} else {
		return requestsMilliCores
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
