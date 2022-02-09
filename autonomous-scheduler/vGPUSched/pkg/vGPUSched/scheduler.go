package vGPUSched

import (
	"context"
	"errors"
	"fmt"

	v1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/client-go/kubernetes"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/klog"

	"github/AI-SIG/alnair-device-plugin/pkg/devicepluginserver"
	"github/AI-SIG/autonomous-scheduler/vGPUSched/pkg/utils"
)

const (
	Name = "vGPUSched"
)

var (
	_ framework.FilterPlugin    = &vGPUSched{}
	_ framework.ScorePlugin     = &vGPUSched{}
    _ framework.ScoreExtensions = &vGPUSched{}
)

type vGPUSched struct {
    handle framework.Handle
}

func New(_ runtime.Object, handle framework.Handle) (framework.Plugin, error) {
    klog.InfoS("Creating new vGPU Scheduling plugin")
	return &vGPUSched{
		handle: handle,
	}, nil
}

func (g *vGPUSched) Name() string {
	return Name
}

func (g *vGPUSched) Filter(ctx context.Context, state *framework.CycleState, pod *v1.Pod, node *framework.NodeInfo) *framework.Status {
	klog.Infof("filter pod: %v, node: %v\n", pod.Name, node.Node().Name)
	nodeinfos := utils.newNodeInfo(node)
	if allocatable := nodeinfos.Assume(pod); allocatable {
	    return framework.NewStatus(framework.Success, "")
	}
	return framework.NewStatus(framework.Unschedulable, "Node:"+node.Node().Name)
}

func (g *vGPUSched) Score(ctx context.Context, state *framework.CycleState, p *v1.Pod, nodeName string) (int64, *framework.Status) {
	// Get Node Info
	nodeInfo, err := g.handle.SnapshotSharedLister().NodeInfos().Get(nodeName)
	if err != nil {
		return 0, framework.NewStatus(framework.Error, fmt.Sprintf("getting node %q from Snapshot: %v", nodeName, err))
	}

    intScore, err := CalculateScore(nodeInfo)
	if err != nil {
		klog.Errorf("CalculateScore Error: %v", err)
		return 0, framework.NewStatus(framework.Error, fmt.Sprintf("Score Node: %v Error: %v", nodeInfo.Node().Name, err))
	}

    return intScore, framework.NewStatus(framework.Success)
}

func CalculateScore(info *framework.NodeInfo) uint64 {
    allocateMemorySum := uint64(0)
    for _, pod := range info.Pods {
		if mem, ok := pod.Pod.GetLabels()["alnair/vgpu-mem"];ok {
			allocateMemorySum += StrToUint64(mem)
		}
	}
	return allocateMemorySum
}

func StrToUint64(str string) uint64 {
	if i, e := strconv.Atoi(str); e != nil {
		return 0
	} else {
		return uint64(i)
	}
}


func (g *vGPUSched) NormalizeScore(_ context.Context, _ *framework.CycleState, pod *v1.Pod, scores framework.NodeScoreList) *framework.Status {
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
		klog.Infof("Node: %v, Score: %v in Plugin: when scheduling Pod: %v/%v", scores[i].Name, scores[i].Score, pod.GetNamespace(), pod.GetName())
	}
	return framework.NewStatus(framework.Success)
}

func (g *vGPUSched) ScoreExtensions() framework.ScoreExtensions {
	return g
}

func (g *vGPUSched) Reserve(ctx context.Context, cycleState *framework.CycleState, pod *v1.Pod, nodeName string) *framework.Status {
    klog.Infof("Reserve success: pod %s on node %s", pod.Name, nodeName)
    // PatchPodAnnotation according to the deviceplugin format
	return framework.NewStatus(framework.Success)

}