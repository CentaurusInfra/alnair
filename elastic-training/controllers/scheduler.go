/*
Copyright github.com/futurewei-cloud.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package controllers

import (
	"context"
	aiv1alpha1 "elastictraining/api/v1alpha1"
	"fmt"
	"time"

	"github.com/go-logr/logr"
	"github.com/prometheus/common/log"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// SchedulerReconciler sets the TargetReplicas of an ElasticHorovodJob object
type SchedulerReconciler struct {
	client.Client
	Log    logr.Logger
	Scheme *runtime.Scheme
}

func (r *SchedulerReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := r.Log.WithValues("ElasticHorovodJob", req.NamespacedName)
	log.Info("Start scheduling")

	var ehjob aiv1alpha1.ElasticHorovodJob
	if err := r.Get(ctx, req.NamespacedName, &ehjob); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	if ehjob.Spec.WorkersSpec.TargetReplicas != nil {
		log.Info("Target replicas already specified, do nothing")
		return ctrl.Result{}, nil
	}

	var nodeList corev1.NodeList
	_ = r.List(ctx, &nodeList)

	var totalGPU int32 = 0
	for _, node := range nodeList.Items {
		gpus := node.Status.Allocatable["nvidia.com/gpu"]
		gpuQuantity, ok := gpus.AsInt64()
		if ok {
			totalGPU += int32(gpuQuantity)
		}
	}

	var podList corev1.PodList
	listOpts := []client.ListOption{client.InNamespace(ehjob.Namespace)}
	_ = r.List(ctx, &podList, listOpts...)

	var usedGPU int32 = 0
	for _, pod := range podList.Items {
		for _, ctnr := range pod.Spec.Containers {
			reqGPUs := ctnr.Resources.Limits["nvidia.com/gpu"]
			reqGPUQuantity, ok := reqGPUs.AsInt64()
			if ok {
				usedGPU += int32(reqGPUQuantity)
			}
		}
	}

	freeGPU := totalGPU - usedGPU
	var targetReplicas int32 = 0
	if freeGPU >= *ehjob.Spec.WorkersSpec.MaxReplicas {
		targetReplicas = *ehjob.Spec.WorkersSpec.MaxReplicas
	} else if freeGPU >= *ehjob.Spec.WorkersSpec.MinReplicas {
		targetReplicas = freeGPU
	}

	// there are enough free GPU to allocate for this job
	if targetReplicas > 0 {
		log.Info(fmt.Sprintf("found enough GPUs, start updating target replicas to be %d", targetReplicas))
		if err := r.updateTargetReplicas(ctx, ehjob, targetReplicas); err != nil {
			return ctrl.Result{RequeueAfter: 5 * time.Second}, nil
		}
		return ctrl.Result{}, nil
	}

	// there are not enough free GPU to accommodate for this job
	var ehjobList aiv1alpha1.ElasticHorovodJobList
	_ = r.List(ctx, &ehjobList, listOpts...)

	var totalPreempt int32 = 0
	for _, ehj := range ehjobList.Items {
		if ehj.Spec.WorkersSpec.TargetReplicas == nil ||
			*ehj.Spec.WorkersSpec.TargetReplicas <= *ehj.Spec.WorkersSpec.MinReplicas {
			continue
		}

		log.Info(fmt.Sprintf("Found preemptable ElasticHorovodJob: %s/%s, target/min Replicas: %d/%d",
			ehj.Namespace, ehj.Name, *ehj.Spec.WorkersSpec.TargetReplicas, *ehj.Spec.WorkersSpec.MinReplicas))

		toPreempt := *ehj.Spec.WorkersSpec.TargetReplicas - *ehj.Spec.WorkersSpec.MinReplicas

		if err := r.updateTargetReplicas(ctx, ehj, *ehj.Spec.WorkersSpec.MinReplicas); err != nil {
			continue
		}

		log.Info(fmt.Sprintf("Preempted ElasticHorovodJob: %s/%s, taget replicas decreased to %d",
			ehj.Namespace, ehj.Name, *ehj.Spec.WorkersSpec.MinReplicas))

		totalPreempt += toPreempt

		if totalPreempt+freeGPU >= *ehjob.Spec.WorkersSpec.MinReplicas {
			// It's greedy to update the target replicas right away. If the update is not successful, the requeue would
			// lead to waiting for enough GPUs by counting all pods requests. Behavior is different between successful and failed updates.
			// Race conditions may happen if another job arrives while this job might be put behind. Need a binding mechanism?
			log.Info(fmt.Sprintf("found enough GPUs through preemption, start updating target replicas to be %d", totalPreempt+freeGPU))
			if err := r.updateTargetReplicas(ctx, ehjob, totalPreempt+freeGPU); err != nil {
				return ctrl.Result{RequeueAfter: 5 * time.Second}, nil
			}
			return ctrl.Result{}, nil
		}
	}

	// cannot find enough GPU even after preemption, wait for some time and try again
	return ctrl.Result{RequeueAfter: 10 * time.Second}, nil
}

func (r *SchedulerReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&aiv1alpha1.ElasticHorovodJob{}).
		Complete(r)
}

func (r *SchedulerReconciler) updateTargetReplicas(ctx context.Context, ehjob aiv1alpha1.ElasticHorovodJob, target int32) error {
	ehjob.Spec.WorkersSpec.TargetReplicas = &target
	if err := r.Update(ctx, &ehjob); err != nil {
		log.Info(fmt.Sprintf("Error in updating target replicas for ElasticHorovodJob %s/%s: %s.",
			ehjob.Namespace, ehjob.Name, err.Error()))
		return err
	}
	return nil
}
