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
	"strings"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	aiv1alpha1 "elastictraining/api/v1alpha1"

	sigsv1alpha1 "sigs.k8s.io/scheduler-plugins/pkg/apis/scheduling/v1alpha1"
)

type ElasticHorovodJobController struct {
	svcName    string
	jobName    string
	workerName string
	pgName     string
}

func (r ElasticHorovodJobController) Test() string {
	return "ElasticHorovodJobController"
}

func (r ElasticHorovodJobController) UpdateStatus(reconciler *UnifiedJobReconciler, ctx context.Context, ujob aiv1alpha1.UnifiedJob, applyOpts []client.PatchOption) (bool, error) {
	jobName := fmt.Sprintf(r.jobName, ujob.Name)
	oldStatus := ujob.Status.UnifiedJobStatus
	var newStatus aiv1alpha1.UnifiedJobStatusType

	//check statefulset status
	if ujob.Spec.ReplicaSpec.TargetReplicas == nil {
		newStatus = aiv1alpha1.JobWaiting
	} else {
		job, err := r.getJob(reconciler, ctx, jobName, ujob.Namespace)
		if err != nil {
			if errors.IsNotFound(err) {
				newStatus = aiv1alpha1.JobPending
				reconciler.Log.Info("Job not found (not ready).")
			} else {
				reconciler.Log.Info(fmt.Sprintf("Error in querying workers: %s.", err.Error()))
			}
		} else {
			if job.Status.Succeeded == 1 {
				newStatus = aiv1alpha1.JobCompleted
			} else if job.Status.Failed == 1 {
				newStatus = aiv1alpha1.JobFailed
			} else if job.Status.Active == 1 {
				newStatus = aiv1alpha1.JobRunning
			}
		}
	}

	changed := newStatus == oldStatus
	ujob.Status.UnifiedJobStatus = newStatus

	return changed, reconciler.Status().Update(ctx, &ujob)

}

func (r ElasticHorovodJobController) ReleaseResources(reconciler *UnifiedJobReconciler, ctx context.Context, ujob aiv1alpha1.UnifiedJob, deleteOpts []client.DeleteOption) error {
	//TODO
	jobName := fmt.Sprintf(r.jobName, ujob.Name)
	workersName := fmt.Sprintf(r.workerName, ujob.Name)

	if err := r.deleteJob(reconciler, ctx, jobName, ujob.Namespace, deleteOpts); err != nil {
		return err
	}

	if err := r.deleteWorkers(reconciler, ctx, workersName, ujob.Namespace, deleteOpts); err != nil {
		return err
	}

	return nil
}

func (r ElasticHorovodJobController) DeleteAll(reconciler *UnifiedJobReconciler, ctx context.Context, ujob aiv1alpha1.UnifiedJob, deleteOpts []client.DeleteOption) error {
	//same as releaseresources + deleteservice
	svcName := fmt.Sprintf(r.svcName, ujob.Name)
	pgName := fmt.Sprintf(r.pgName, ujob.Name)

	if err := r.deleteService(reconciler, ctx, svcName, ujob.Namespace, deleteOpts); err != nil {
		reconciler.Log.Info(fmt.Sprintf("Error in deleting service: %s", err.Error()))
	}

	if err := r.deletePodGroup(reconciler, ctx, pgName, ujob.Namespace, deleteOpts); err != nil {
		reconciler.Log.Info(fmt.Sprintf("Error in deleting pod group: %s", err.Error()))
	}

	return r.ReleaseResources(reconciler, ctx, ujob, deleteOpts)
}

func (r ElasticHorovodJobController) ServiceExists(reconciler *UnifiedJobReconciler, ctx context.Context, ujob aiv1alpha1.UnifiedJob) bool {
	var svc corev1.Service
	svcName := fmt.Sprintf(r.svcName, ujob.Name)
	key := types.NamespacedName{
		Namespace: ujob.Namespace,
		Name:      svcName,
	}

	if err := reconciler.Get(ctx, key, &svc); err != nil {
		return false
	}

	return true
}

func (r ElasticHorovodJobController) CreateService(reconciler *UnifiedJobReconciler, ctx context.Context, applyOpts []client.PatchOption, ujob aiv1alpha1.UnifiedJob) error {
	svc, err := r.desiredService(reconciler, ctx, ujob)
	if err != nil {
		//reconciler.Log.Info(fmt.Sprintf("Error in creating service: %s", err))
		return err
	}

	if err := reconciler.Patch(ctx, &svc, client.Apply, applyOpts...); err != nil {
		return err
	}

	pg, err := r.desiredPodGroup(reconciler, ctx, ujob)
	if err != nil {
		//reconciler.Log.Info(fmt.Sprintf("Error in creating pod group: %s", err))
		return err
	}

	if err := reconciler.Patch(ctx, &pg, client.Apply, applyOpts...); err != nil {
		return err
	}

	return nil
}

func (r ElasticHorovodJobController) StuckInPending(reconciler *UnifiedJobReconciler, ctx context.Context, ujob aiv1alpha1.UnifiedJob) bool {

	passed := false

	for i := 1; i <= 2; i++ {
		podList, err := r.getSSPods(reconciler, ctx, ujob.Name, ujob.Namespace)

		if err != nil {
			if errors.IsNotFound(err) {
				return false
			}
			reconciler.Log.Info(fmt.Sprintf("Could not find workers: %s", err.Error()))
			return false
		}

		for _, pod := range podList {
			if pod.Status.Phase == corev1.PodPending {
				if !passed {
					time.Sleep(10 * time.Second)
					passed = true
					break
				}
				return true
			}
		}

		if !passed {
			return false
		}

	}

	return false //unreachable

}

func (r ElasticHorovodJobController) PatchAll(reconciler *UnifiedJobReconciler, ctx context.Context, ujob aiv1alpha1.UnifiedJob, applyOpts []client.PatchOption) error {
	//create and patch the job
	// TODO: Complete
	m := ujob.Spec.ReplicaSpec.TargetReplicas

	var numGpu int64
	var nodeName string
	for k, v := range m {
		numGpu = v
		nodeName = k
	}

	ss, err := r.desiredWorkers(reconciler, ctx, ujob, numGpu, nodeName)
	if err != nil {
		reconciler.Log.Info(fmt.Sprintf("Error in creating workers: %s.", err.Error()))
		return err
	}

	if err := reconciler.Patch(ctx, &ss, client.Apply, applyOpts...); err != nil {
		reconciler.Log.Info(fmt.Sprintf("Error in patching workers: %s.", err.Error()))
		return err
	}

	//TODO: add breaks to check if desired workers are ready

	job, err := r.desiredJob(reconciler, ctx, ujob, numGpu, nodeName)
	if err != nil {
		reconciler.Log.Info(fmt.Sprintf("Error in creating job: %s.", err.Error()))
		return err
	}

	if err := reconciler.Patch(ctx, &job, client.Apply, applyOpts...); err != nil {
		reconciler.Log.Info(fmt.Sprintf("Error in patching job: %s.", err.Error()))
		return err
	}

	return nil
}

//DESIRED OBJECTS ------------------------------------------------------

func (r ElasticHorovodJobController) desiredService(reconciler *UnifiedJobReconciler, ctx context.Context, ujob aiv1alpha1.UnifiedJob) (corev1.Service, error) {
	svcName := fmt.Sprintf(r.svcName, ujob.Name)
	svc := corev1.Service{
		TypeMeta: metav1.TypeMeta{APIVersion: corev1.SchemeGroupVersion.String(), Kind: "Service"},
		ObjectMeta: metav1.ObjectMeta{
			Name:      svcName,
			Namespace: ujob.Namespace,
		},
		Spec: corev1.ServiceSpec{
			ClusterIP: "None",
			Selector:  map[string]string{"UnifiedJob": ujob.Name, "role": "worker"},
		},
	}

	if err := ctrl.SetControllerReference(&ujob, &svc, reconciler.Scheme); err != nil {
		return svc, err
	}

	return svc, nil
}

func (r ElasticHorovodJobController) desiredPodGroup(reconciler *UnifiedJobReconciler, ctx context.Context, ujob aiv1alpha1.UnifiedJob) (sigsv1alpha1.PodGroup, error) {
	pgName := fmt.Sprintf(r.pgName, ujob.Name)

	pg := sigsv1alpha1.PodGroup{
		TypeMeta: metav1.TypeMeta{APIVersion: "scheduling.sigs.k8s.io/v1alpha1", Kind: "PodGroup"},
		ObjectMeta: metav1.ObjectMeta{
			Name:      pgName,
			Namespace: ujob.Namespace,
		},
		Spec: sigsv1alpha1.PodGroupSpec{
			MinMember: 0, //*ehjob.Spec.WorkersSpec.MinReplicas
			MinResources: &corev1.ResourceList{
				corev1.ResourceName("nvidia.com/gpu"): *resource.NewQuantity(0, resource.DecimalSI),
			},
		},
		Status: sigsv1alpha1.PodGroupStatus{
			ScheduleStartTime: metav1.Now(),
		},
	}

	if err := ctrl.SetControllerReference(&ujob, &pg, reconciler.Scheme); err != nil {
		return pg, err
	}

	return pg, nil
}

func (r ElasticHorovodJobController) desiredJob(reconciler *UnifiedJobReconciler, ctx context.Context, ujob aiv1alpha1.UnifiedJob, numGpu int64, nodeName string) (batchv1.Job, error) {
	jobName := fmt.Sprintf(r.jobName, ujob.Name)
	svcName := fmt.Sprintf(r.svcName, ujob.Name)
	//podGroupName := fmt.Sprintf("unifiedjobpodgroup-%s", ujob.Name)

	sshkeysMode := int32(0600)
	scriptMode := int32(0744)
	one := int32(1)
	sshSetupCommand := "mkdir -p /root/.ssh; cp /etc/secrets/* /root/.ssh/; chmod 644 /root/.ssh/authorized_keys;"
	scriptSetupCmd := fmt.Sprintf("mkdir -p /scripts; cp /etc/scripts/* /scripts/; sed -i 's/SERVICENAME/%s/' /scripts/discover_hosts.sh;", svcName)
	horovodArgs := fmt.Sprintf("-np %d --max-np %d --host-discovery-script /scripts/discover_hosts.sh",
		*ujob.Spec.ReplicaSpec.MinReplicas,
		*ujob.Spec.ReplicaSpec.MaxReplicas)
	pythonCommand := strings.Join(ujob.Spec.JobSpec.UnifiedArgs, " ")
	wholeCommand := fmt.Sprintf("%s %s horovodrun -p 12345 %s %s", sshSetupCommand, scriptSetupCmd, horovodArgs, pythonCommand)

	job := batchv1.Job{
		TypeMeta: metav1.TypeMeta{APIVersion: batchv1.SchemeGroupVersion.String(), Kind: "Job"},
		ObjectMeta: metav1.ObjectMeta{
			Name:      jobName,
			Namespace: ujob.Namespace,
		},
		Spec: batchv1.JobSpec{
			Template: corev1.PodTemplateSpec{
				//commented because podgroup only supports single-object podgroup (podgroup
				//label is enabled in worker statefulset)

				//ObjectMeta: metav1.ObjectMeta{
				// 	Labels: map[string]string{
				// 		"pod-group.scheduling.sigs.k8s.io": podGroupName,
				// 	},
				// },
				Spec: corev1.PodSpec{
					SchedulerName: "default-scheduler", //change if adding additional schedulers
					Volumes: []corev1.Volume{
						{
							Name: "sshkeys",
							VolumeSource: corev1.VolumeSource{
								Secret: &corev1.SecretVolumeSource{
									SecretName:  "horovod-sshkeys",
									DefaultMode: &sshkeysMode,
								},
							},
						},
						{
							Name: "scripts",
							VolumeSource: corev1.VolumeSource{
								ConfigMap: &corev1.ConfigMapVolumeSource{
									LocalObjectReference: corev1.LocalObjectReference{
										Name: "ai-horovod-discover-hosts",
									},
									DefaultMode: &scriptMode,
								},
							},
						},
					},
					Containers: []corev1.Container{
						{
							Name:  "launcher",
							Image: ujob.Spec.JobSpec.Image,
							VolumeMounts: []corev1.VolumeMount{
								{
									Name:      "sshkeys",
									MountPath: "/etc/secrets",
								},
								{
									Name:      "scripts",
									MountPath: "/etc/scripts",
								},
							},
							Command: []string{"/bin/sh"},
							Args: []string{
								"-c",
								wholeCommand,
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

func (r ElasticHorovodJobController) desiredWorkers(reconciler *UnifiedJobReconciler, ctx context.Context, ujob aiv1alpha1.UnifiedJob, numGpus int64, nodeName string) (appsv1.StatefulSet, error) {
	//TODO: for now it is just a single node
	//TODO: patch podgroup as well
	defaultMode := int32(0600)
	numGpus2 := int32(numGpus)
	workersName := fmt.Sprintf(r.workerName, ujob.Name)
	svcName := fmt.Sprintf(r.svcName, ujob.Name)
	pgName := fmt.Sprintf(r.pgName, ujob.Name)
	statefulset := appsv1.StatefulSet{
		TypeMeta: metav1.TypeMeta{APIVersion: appsv1.SchemeGroupVersion.String(), Kind: "StatefulSet"},
		ObjectMeta: metav1.ObjectMeta{
			Name:      workersName,
			Namespace: ujob.Namespace,
		},
		Spec: appsv1.StatefulSetSpec{
			ServiceName:         svcName,
			Replicas:            &numGpus2,
			PodManagementPolicy: appsv1.PodManagementPolicyType(string(appsv1.ParallelPodManagement)),
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{"UnifiedJob": ujob.Name, "role": "worker"},
			},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{ //add labels such as podgroup # and statefulset
					Labels: map[string]string{"UnifiedJob": ujob.Name, "role": "worker",
						"pod-group.scheduling.sigs.k8s.io": pgName,
					},
				},
				Spec: corev1.PodSpec{
					SchedulerName: "default-scheduler", //change if adding other schedulers
					Volumes: []corev1.Volume{
						{
							Name: "sshkeys",
							VolumeSource: corev1.VolumeSource{
								Secret: &corev1.SecretVolumeSource{
									SecretName:  "horovod-sshkeys",
									DefaultMode: &defaultMode,
								},
							},
						},
						//shared memory
						{
							Name: "dshm",
							VolumeSource: corev1.VolumeSource{
								EmptyDir: &corev1.EmptyDirVolumeSource{
									Medium: "Memory",
								},
							},
						},
					},
					Containers: []corev1.Container{
						{
							Name:  "worker",
							Image: ujob.Spec.JobSpec.Image,
							VolumeMounts: []corev1.VolumeMount{
								{
									Name:      "sshkeys",
									MountPath: "/etc/secrets",
								},
								//add shared memory
								{
									Name:      "dshm",
									MountPath: "/dev/shm",
								},
							},
							Command: []string{"/bin/sh"},
							Args: []string{
								"-c",
								"/usr/sbin/sshd -p 12345; mkdir -p /root/.ssh; cp /etc/secrets/* /root/.ssh/; chmod 644 /root/.ssh/authorized_keys; sleep infinity",
							},
							Resources: corev1.ResourceRequirements{
								Limits: corev1.ResourceList{
									corev1.ResourceName("nvidia.com/gpu"): *resource.NewQuantity(1, resource.DecimalSI),
								},
							},
						},
					},
				},
			},
		},
	}

	if err := ctrl.SetControllerReference(&ujob, &statefulset, reconciler.Scheme); err != nil {
		return statefulset, err
	}

	return statefulset, nil
}

//GET OBJECTS	------------------------------------------------------

func (r ElasticHorovodJobController) getService(reconciler *UnifiedJobReconciler, ctx context.Context, name string, namespace string) (corev1.Service, error) {
	var svc corev1.Service
	key := types.NamespacedName{
		Namespace: namespace,
		Name:      name,
	}

	if err := reconciler.Get(ctx, key, &svc); err != nil {
		return svc, err
	}

	return svc, nil
}

func (r ElasticHorovodJobController) getJob(reconciler *UnifiedJobReconciler, ctx context.Context, name string, namespace string) (batchv1.Job, error) {
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

func (r ElasticHorovodJobController) getWorkers(reconciler *UnifiedJobReconciler, ctx context.Context, name string, namespace string) (appsv1.StatefulSet, error) {
	var ss appsv1.StatefulSet
	key := types.NamespacedName{
		Namespace: namespace,
		Name:      name,
	}

	if err := reconciler.Get(ctx, key, &ss); err != nil {
		return ss, err
	}

	return ss, nil
}

func (r ElasticHorovodJobController) getPodGroup(reconciler *UnifiedJobReconciler, ctx context.Context, name string, namespace string) (sigsv1alpha1.PodGroup, error) {
	var pg sigsv1alpha1.PodGroup
	key := types.NamespacedName{
		Namespace: namespace,
		Name:      name,
	}

	if err := reconciler.Get(ctx, key, &pg); err != nil {
		return pg, err
	}

	return pg, nil
}

func (r ElasticHorovodJobController) getSSPods(reconciler *UnifiedJobReconciler, ctx context.Context, name string, namespace string) ([]corev1.Pod, error) {
	labels := map[string]string{"UnifiedJob": name, "role": "worker"}
	return reconciler.getPodsByLabel(ctx, namespace, labels)
}

//DELETE OBJECTS------------------------------------------------------

func (r ElasticHorovodJobController) deleteService(reconciler *UnifiedJobReconciler, ctx context.Context, name, namespace string, deleteOpts []client.DeleteOption) error {

	svc, err := r.getService(reconciler, ctx, name, namespace)
	if err != nil {
		return nil
	}

	return reconciler.Delete(ctx, &svc, deleteOpts...)
}

func (r ElasticHorovodJobController) deleteJob(reconciler *UnifiedJobReconciler, ctx context.Context, name, namespace string, deleteOpts []client.DeleteOption) error {

	job, err := r.getJob(reconciler, ctx, name, namespace)
	if err != nil {
		return nil
	}

	return reconciler.Delete(ctx, &job, deleteOpts...)
}

func (r ElasticHorovodJobController) deleteWorkers(reconciler *UnifiedJobReconciler, ctx context.Context, name, namespace string, deleteOpts []client.DeleteOption) error {

	workers, err := r.getWorkers(reconciler, ctx, name, namespace)
	if err != nil {
		return nil
	}

	return reconciler.Delete(ctx, &workers, deleteOpts...)
}

func (r ElasticHorovodJobController) deletePodGroup(reconciler *UnifiedJobReconciler, ctx context.Context, name, namespace string, deleteOpts []client.DeleteOption) error {

	pg, err := r.getJob(reconciler, ctx, name, namespace)
	if err != nil {
		return nil
	}

	return reconciler.Delete(ctx, &pg, deleteOpts...)
}
