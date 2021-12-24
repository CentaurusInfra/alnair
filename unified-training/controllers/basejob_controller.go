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
	errors2 "errors"
	"fmt"
	"strings"
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	"sigs.k8s.io/controller-runtime/pkg/client"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
)

type BaseJobController struct {
	jobName string
}

func (r BaseJobController) Test() string {
	return "BaseJobController"
}

func (r BaseJobController) UpdateStatus(reconciler *UnifiedJobReconciler, ctx context.Context, ujob aiv1alpha1.UnifiedJob, applyOpts []client.PatchOption) (bool, error) {
	jobName := fmt.Sprintf("unifiedjob-%s", ujob.Name)
	oldStatus := ujob.Status.UnifiedJobStatus
	var newStatus aiv1alpha1.UnifiedJobStatusType

	if ujob.Spec.ReplicaSpec.TargetReplicas == nil {
		newStatus = aiv1alpha1.JobWaiting
	} else if len(ujob.Spec.ReplicaSpec.TargetReplicas) > 1 {
		newStatus = aiv1alpha1.JobFailed
		reconciler.Log.Info("Target Replicas is incorrect; stay in pending until restart")
	} else {
		job, err := r.getJob(reconciler, ctx, jobName, ujob.Namespace)
		if err != nil {
			if !errors.IsNotFound(err) {
				reconciler.Log.Info(fmt.Sprintf("Error in querying job: %s.", err.Error()))
			}
			newStatus = aiv1alpha1.JobPending
		} else {
			pod, err := r.getJobPod(reconciler, ctx, job)
			if err != nil {
				if err.Error() != "IsNotFound" {
					reconciler.Log.Info(fmt.Sprintf("Error in querying workers: %s.", err.Error()))
				}
				return false, nil
			}

			if podPending(pod) {
				newStatus = aiv1alpha1.JobPending
			} else if pod.Status.Phase == corev1.PodRunning {
				newStatus = aiv1alpha1.JobRunning
			} else if pod.Status.Phase == corev1.PodFailed {
				if pod.Status.Reason == "UnexpectedAdmissionError" {
					newStatus = aiv1alpha1.JobPending
				} else {
					newStatus = aiv1alpha1.JobFailed
				}
			} else if pod.Status.Phase == corev1.PodSucceeded {
				newStatus = aiv1alpha1.JobCompleted
			}

		}
	}

	changed := newStatus == oldStatus
	ujob.Status.UnifiedJobStatus = newStatus

	return changed, reconciler.Status().Update(ctx, &ujob)

}

func (r BaseJobController) ReleaseResources(reconciler *UnifiedJobReconciler, ctx context.Context, ujob aiv1alpha1.UnifiedJob, deleteOpts []client.DeleteOption) error {
	jobName := fmt.Sprintf("unifiedjob-%s", ujob.Name)
	if err := r.deleteJob(reconciler, ctx, jobName, ujob.Namespace, deleteOpts); err != nil {
		return err
	}
	return nil
}

func (r BaseJobController) DeleteAll(reconciler *UnifiedJobReconciler, ctx context.Context, ujob aiv1alpha1.UnifiedJob, deleteOpts []client.DeleteOption) error {
	return r.ReleaseResources(reconciler, ctx, ujob, deleteOpts)
}

func (r BaseJobController) ServiceExists(reconciler *UnifiedJobReconciler, ctx context.Context, ujob aiv1alpha1.UnifiedJob) bool {
	return true
}

func (r BaseJobController) CreateService(reconciler *UnifiedJobReconciler, ctx context.Context, applyOpts []client.PatchOption, ujob aiv1alpha1.UnifiedJob) error {
	return nil
}

func (r BaseJobController) StuckInPending(reconciler *UnifiedJobReconciler, ctx context.Context, ujob aiv1alpha1.UnifiedJob) bool {

	job, err := r.getJob(reconciler, ctx, fmt.Sprintf(r.jobName, ujob.Name), ujob.Namespace)

	if err != nil {
		if errors.IsNotFound(err) {
			reconciler.Log.Info("Job not found.")
		} else {
			reconciler.Log.Info(fmt.Sprintf("Job error: %s", err))
		}
		return false
	}

	passed := false

	for i := 1; i <= 2; i++ {

		pod, err := r.getJobPod(reconciler, ctx, job)

		if err != nil {
			reconciler.Log.Info(fmt.Sprintf("Could not find job pod: %s", err.Error()))
			return false
		}

		if podPending(pod) {
			if !passed {
				time.Sleep(10 * time.Second)
				passed = true
				break
			}
			return true
		}

		if !passed {
			return false
		}
	}

	return false

}

func (r BaseJobController) PatchAll(reconciler *UnifiedJobReconciler, ctx context.Context, ujob aiv1alpha1.UnifiedJob, applyOpts []client.PatchOption) error {
	//create and patch the job
	m := ujob.Spec.ReplicaSpec.TargetReplicas

	var numGpu int64
	var nodeName string
	for k, v := range m {
		numGpu = v
		nodeName = k
	}

	job, err := r.desiredJob(reconciler, ctx, ujob, numGpu, nodeName)
	if err != nil {
		reconciler.Log.Info(fmt.Sprintf("Error in creating job: %s.", err.Error()))
		return err
	}

	err = reconciler.Patch(ctx, &job, client.Apply, applyOpts...)
	if err != nil {
		reconciler.Log.Info(fmt.Sprintf("Error in patching job: %s.", err.Error()))
		return err
	}

	return nil
}

func (r BaseJobController) desiredJob(reconciler *UnifiedJobReconciler, ctx context.Context, ujob aiv1alpha1.UnifiedJob, numGpu int64, nodeName string) (batchv1.Job, error) {

	one := int32(1)
	command := strings.Join(ujob.Spec.JobSpec.UnifiedArgs, " ")

	job := batchv1.Job{
		TypeMeta: metav1.TypeMeta{APIVersion: batchv1.SchemeGroupVersion.String(), Kind: "Job"},
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("unifiedjob-%s", ujob.Name),
			Namespace: ujob.Namespace,
		},
		Spec: batchv1.JobSpec{
			Template: corev1.PodTemplateSpec{

				Spec: corev1.PodSpec{
					SchedulerName: "default-scheduler", //change if adding additional schedulers
					Containers: []corev1.Container{
						{
							Name:    "job",
							Image:   ujob.Spec.JobSpec.Image,
							Command: []string{"/bin/sh"},
							Args: []string{
								"-c",
								command,
							},
							Resources: corev1.ResourceRequirements{
								Limits: corev1.ResourceList{
									corev1.ResourceName("nvidia.com/gpu"): *resource.NewQuantity(numGpu, resource.DecimalSI),
								},
							},
						},
					},
					RestartPolicy: corev1.RestartPolicyNever,
					NodeName:      nodeName,
				},
			},
			BackoffLimit: &one,
		},
	}

	if err := ctrl.SetControllerReference(&ujob, &job, reconciler.Scheme); err != nil {
		return job, err
	}

	return job, nil
}

func (r BaseJobController) getJob(reconciler *UnifiedJobReconciler, ctx context.Context, name string, namespace string) (batchv1.Job, error) {
	var job batchv1.Job
	key := types.NamespacedName{
		Namespace: namespace,
		Name:      name,
	}

	if err := reconciler.Get(ctx, key, &job); err != nil {
		return job, err
	}

	return job, nil
}

func (r BaseJobController) getJobPod(reconciler *UnifiedJobReconciler, ctx context.Context, job batchv1.Job) (corev1.Pod, error) {
	labels := map[string]string{
		"job-name": job.Name, //kubernetes auto-generated
	}

	podList, err := reconciler.getPodsByLabel(ctx, job.Namespace, labels)
	if len(podList) == 0 {
		var pod corev1.Pod
		return pod, errors2.New("IsNotFound")
	}
	return podList[0], err
}

func (r BaseJobController) deleteJob(reconciler *UnifiedJobReconciler, ctx context.Context, name, namespace string, deleteOpts []client.DeleteOption) error {
	var workers batchv1.Job
	key := types.NamespacedName{
		Namespace: namespace,
		Name:      name,
	}

	if err := reconciler.Get(ctx, key, &workers); err != nil {
		return nil
	}

	return reconciler.Delete(ctx, &workers, deleteOpts...)
}
