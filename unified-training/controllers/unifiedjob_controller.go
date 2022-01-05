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
	"fmt"
	"time"

	"github.com/go-logr/logr"
	appsv1 "k8s.io/api/apps/v1"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	aiv1alpha1 "elastictraining/api/v1alpha1"
)

// UnifiedJobReconciler reconciles a UnifiedJob object
type UnifiedJobReconciler struct {
	client.Client
	Log             logr.Logger
	Scheme          *runtime.Scheme
	JobInterfaceMap map[aiv1alpha1.UnifiedJobType]UnifiedJobInterface
}

type UnifiedJobInterface interface {
	//test simply ensures that the interface is plugged in correctly
	Test() string

	//update status of UnifiedJob.Status.UnifiedJobStatus
	//returns bool for true if status changed, false elsewise
	UpdateStatus(reconciler *UnifiedJobReconciler, ctx context.Context, ujob aiv1alpha1.UnifiedJob, applyOpts []client.PatchOption) (bool, error)

	//delete job, release resources, but not necessarily wipe everything out
	ReleaseResources(reconciler *UnifiedJobReconciler, ctx context.Context, ujob aiv1alpha1.UnifiedJob, deleteOpts []client.DeleteOption) error

	//delete everything, ie resources + service(s)
	DeleteAll(reconciler *UnifiedJobReconciler, ctx context.Context, ujob aiv1alpha1.UnifiedJob, deleteOpts []client.DeleteOption) error

	//check if service exists
	ServiceExists(reconciler *UnifiedJobReconciler, ctx context.Context, ujob aiv1alpha1.UnifiedJob) bool

	//create and patch service
	CreateService(reconciler *UnifiedJobReconciler, ctx context.Context, applyOpts []client.PatchOption, ujob aiv1alpha1.UnifiedJob) error

	//check if job is stuck in pending; ie resource conflict / race condition
	StuckInPending(reconciler *UnifiedJobReconciler, ctx context.Context, ujob aiv1alpha1.UnifiedJob) bool

	//create and patch all
	PatchAll(reconciler *UnifiedJobReconciler, ctx context.Context, ujob aiv1alpha1.UnifiedJob, applyOpts []client.PatchOption) error
}

// +kubebuilder:rbac:groups=ai.centauruscloud.io,resources=Unifiedjobs,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=ai.centauruscloud.io,resources=Unifiedjobs/status,verbs=get;update;patch

func (r *UnifiedJobReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := r.Log.WithValues("UnifiedJob", req.NamespacedName)
	log.Info("Start reconciling")

	var ujob aiv1alpha1.UnifiedJob
	if err := r.Get(ctx, req.NamespacedName, &ujob); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	if r.JobInterfaceMap[ujob.JobType] == nil {
		log.Info(fmt.Sprintf("UnifiedJobType %s is invalid.", ujob.JobType))
		return ctrl.Result{}, nil
	}

	if ujob.Status.UnifiedJobStatus == aiv1alpha1.JobCompleted || ujob.Status.UnifiedJobStatus == aiv1alpha1.JobFailed {
		log.Info("Job is already finished, do not make further changes.")
		return ctrl.Result{}, nil
	}

	jobController := r.JobInterfaceMap[ujob.JobType]

	applyOpts := []client.PatchOption{client.ForceOwnership, client.FieldOwner("unifiedjob-controller")}

	change, err := jobController.UpdateStatus(r, ctx, ujob, applyOpts)

	if err != nil {
		log.Info(fmt.Sprintf("Error in updating status: %s.", err.Error()))
		return ctrl.Result{RequeueAfter: 5 * time.Second}, nil
	}

	//re-get ujob status
	time.Sleep(3 * time.Second)
	if err := r.Get(ctx, req.NamespacedName, &ujob); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	if !change && ujob.Status.UnifiedJobStatus != aiv1alpha1.JobPending {
		return ctrl.Result{}, nil
	}

	if change {
		log.Info(fmt.Sprintf("Job %s updated status to %s", ujob.Name, ujob.Status.UnifiedJobStatus))
	}

	//do nothing if job completed
	if ujob.Status.UnifiedJobStatus == aiv1alpha1.JobWaiting {
		log.Info("Nothing in TargetReplicas")
		return ctrl.Result{}, nil
	}

	grace := int64(10)
	deletepolicy := metav1.DeletePropagationForeground
	deleteOpts := []client.DeleteOption{&client.DeleteOptions{
		GracePeriodSeconds: &grace,
		PropagationPolicy:  &deletepolicy,
	}}

	//check if job completed
	if ujob.Status.UnifiedJobStatus == aiv1alpha1.JobCompleted {
		log.Info("Job has completed.")
		if err := jobController.DeleteAll(r, ctx, ujob, deleteOpts); err != nil {
			log.Info(fmt.Sprintf("Error in cleaning up resources: %s.", err.Error()))
		}
		return ctrl.Result{}, nil
	}

	//may want to treat completed jobs and errored jobs differently
	if ujob.Status.UnifiedJobStatus == aiv1alpha1.JobFailed {
		log.Info("Job has failed.") //upon failure, reset target replicas,
		if err := jobController.DeleteAll(r, ctx, ujob, deleteOpts); err != nil {
			log.Info(fmt.Sprintf("Error in cleaning up resources: %s.", err.Error()))
		}
		if err := r.resetTargetReplicas(ctx, ujob); err != nil {
			log.Info(fmt.Sprintf("Error in resetting target replicas: %s.", err.Error()))
		}
		return ctrl.Result{}, nil
	}

	//create service if necessary
	if !jobController.ServiceExists(r, ctx, ujob) {
		if err := jobController.CreateService(r, ctx, applyOpts, ujob); err != nil {
			log.Info(fmt.Sprintf("Error in creating service: %s.", err.Error()))
			return ctrl.Result{RequeueAfter: 5 * time.Second}, nil
		}
		log.Info("Service successfully created.")
	}

	log.Info("Service Exists")

	//check if job is running as expected
	if ujob.Status.UnifiedJobStatus == aiv1alpha1.JobRunning {
		//end requeues
		log.Info("Job is running as expected.")
		return ctrl.Result{}, nil
	}

	// check if job has conflicts in scheduling (podgroup, stuck in pending)
	if jobController.StuckInPending(r, ctx, ujob) {
		log.Info("Job is stuck in pending due to possible scheduling conflicts; " +
			"releasing resources and resetting TargetReplicas.")
		if err := jobController.ReleaseResources(r, ctx, ujob, deleteOpts); err != nil {
			log.Info(fmt.Sprintf("Error in cleaning up resources: %s.", err.Error()))
		}
		if err := r.resetTargetReplicas(ctx, ujob); err != nil {
			log.Info(fmt.Sprintf("Error in resetting target replicas: %s.", err.Error()))
		}
		time.Sleep(3 * time.Second)
		return ctrl.Result{}, nil
	}

	//check if workers are not ready
	if err := jobController.PatchAll(r, ctx, ujob, applyOpts); err != nil {
		log.Info(fmt.Sprintf("Job was unable to be created; %s", err.Error()))
		return ctrl.Result{RequeueAfter: 5 * time.Second}, nil
	}

	log.Info("Job was successfully created.")

	//requeue to update status
	return ctrl.Result{RequeueAfter: 5 * time.Second}, nil
}

func (r *UnifiedJobReconciler) SetupWithManager(mgr ctrl.Manager) error {
	//basejob_controller := &BaseJobController{}
	var basejob_controller UnifiedJobInterface = BaseJobController{
		jobName: "basejob-%s",
	}
	var torchelastic_controller UnifiedJobInterface = TorchElasticJobController{
		etcdSvcName:    "torchelastic-etcd-service",
		etcdServerName: "etcd",
		jobName:        "%s-epjob",
		jobID:          "%s-epjobid",
		workerName:     "%s-workers-%d",
		workerSvcName:  "%s-workers-service",
		launcherName:   "%s-launcher",
	}
	var elastichorovod_controller UnifiedJobInterface = ElasticHorovodJobController{
		svcName:    "ehservice-%s",
		jobName:    "ehjob-%s",
		workerName: "ehworkers-%s",
		pgName:     "ehpodgroup-%s",
	}

	r.JobInterfaceMap = map[aiv1alpha1.UnifiedJobType]UnifiedJobInterface{
		aiv1alpha1.BasicJobType:          basejob_controller,
		aiv1alpha1.ElasticPytorchJobType: torchelastic_controller,
		aiv1alpha1.ElasticHorovodJobType: elastichorovod_controller,
	}

	return ctrl.NewControllerManagedBy(mgr).
		For(&aiv1alpha1.UnifiedJob{}).
		Owns(&corev1.Service{}).
		Owns(&appsv1.StatefulSet{}).
		Owns(&batchv1.Job{}).
		Complete(r)
}

func (r *UnifiedJobReconciler) resetTargetReplicas(ctx context.Context, ujob aiv1alpha1.UnifiedJob) error {
	ujob.Spec.ReplicaSpec.TargetReplicas = make(map[string]int64)
	return r.Update(ctx, &ujob)
}

//GENERIC HELPER FUNCTIONS -----------------------------------------------

func (r *UnifiedJobReconciler) getPodsByLabel(ctx context.Context, namespace string, labels map[string]string) ([]corev1.Pod, error) {
	var pl corev1.PodList

	listOpts := []client.ListOption{
		client.InNamespace(namespace),
		client.MatchingLabels(labels),
	}

	if err := r.List(ctx, &pl, listOpts...); err != nil {
		return pl.Items, err
	}

	return pl.Items, nil
}

func (r *UnifiedJobReconciler) getService(ctx context.Context, name string, namespace string) (corev1.Service, error) {
	//generic get service (can be used for etcd service and pod service)
	var svc corev1.Service
	key := types.NamespacedName{
		Namespace: namespace,
		Name:      name,
	}

	if err := r.Get(ctx, key, &svc); err != nil {
		return svc, err
	}

	return svc, nil
}

func (r *UnifiedJobReconciler) getPod(ctx context.Context, name string, namespace string) (corev1.Pod, error) {
	//generic get service (can be used for etcd service and pod service)
	var pod corev1.Pod
	key := types.NamespacedName{
		Namespace: namespace,
		Name:      name,
	}

	if err := r.Get(ctx, key, &pod); err != nil {
		return pod, err
	}

	return pod, nil
}

func getCurrentNodeConfig(podList []corev1.Pod) map[string]int64 {
	nodeMap := make(map[string]int64)

	for _, pod := range podList {
		quant := pod.Spec.Containers[0].Resources.Limits[corev1.ResourceName("nvidia.com/gpu")]
		nodeMap[pod.Spec.NodeName] = quant.ToDec().Value()
	}

	return nodeMap
}

func getTotalGPU(nodeMap map[string]int64) int {
	total := 0
	for _, numGpu := range nodeMap {
		total += int(numGpu)
	}
	return total
}

func podPending(pod corev1.Pod) bool {
	//unexpectedadmissionerror = pending
	if pod.Status.Phase == corev1.PodPending {
		return true
	}

	if pod.Status.Phase == corev1.PodFailed && pod.Status.Reason == "UnexpectedAdmissionError" {
		return true
	}

	return false
}
