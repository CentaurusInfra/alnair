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

	"github.com/go-logr/logr"
	appsv1 "k8s.io/api/apps/v1"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	aiv1alpha1 "elastictraining/api/v1alpha1"
)

// HorovodJobReconciler reconciles a HorovodJob object
type HorovodJobReconciler struct {
	client.Client
	Log    logr.Logger
	Scheme *runtime.Scheme
}

// +kubebuilder:rbac:groups=ai.centauruscloud.io,resources=horovodjobs,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=ai.centauruscloud.io,resources=horovodjobs/status,verbs=get;update;patch

func (r *HorovodJobReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := r.Log.WithValues("HorovodJob", req.NamespacedName)
	log.Info("Start reconciling")

	var hjob aiv1alpha1.HorovodJob
	if err := r.Get(ctx, req.NamespacedName, &hjob); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	workersName := fmt.Sprintf("horovodjob-%s-workers", hjob.Name)
	launcherName := fmt.Sprintf("horovodjob-%s-launcher", hjob.Name)

	if hjob.Status.Workers != "deployment.apps/"+workersName ||
		hjob.Status.Launcher != "job.batch/"+launcherName {
		hjob.Status.Workers = "deployment.apps/" + workersName
		hjob.Status.Launcher = "job.batch/" + launcherName

		if err := r.Status().Update(ctx, &hjob); err != nil {
			log.Info(fmt.Sprintf("Error in updating launcher and workers names: %s.", err.Error()))
		}
	}

	workersReady, err := r.areWorkersReady(ctx, workersName, hjob.Namespace)

	if err != nil {
		if !errors.IsNotFound(err) {
			log.Info(fmt.Sprintf("Error in querying workers: %s.", err.Error()))
		}
	}

	var workers appsv1.Deployment
	applyOpts := []client.PatchOption{client.ForceOwnership, client.FieldOwner("horovodjob-controller")}
	if !workersReady {
		workers, err = r.desiredWorkers(hjob, workersName)
		if err != nil {
			return ctrl.Result{}, err
		}

		err = r.Patch(ctx, &workers, client.Apply, applyOpts...)
		if err != nil {
			log.Info(fmt.Sprintf("Error in patching workers: %s.", err.Error()))
			return ctrl.Result{Requeue: true}, nil
		}

		if isReady, err := r.areWorkersReady(ctx, workers.Name, workers.Namespace); err != nil {
			log.Info(fmt.Sprintf("Error in workers: %s.", err.Error()))
			return ctrl.Result{Requeue: true}, nil
		} else {
			if !isReady {
				log.Info("workers are not ready yet.")
				return ctrl.Result{}, nil
			}
		}
	}

	// Now workers are ready
	if hjob.Status.AvailableWorkerReplicas != *hjob.Spec.WorkersSpec.Replicas {
		hjob.Status.AvailableWorkerReplicas = *hjob.Spec.WorkersSpec.Replicas

		if err := r.Status().Update(ctx, &hjob); err != nil {
			log.Info(fmt.Sprintf("Error in updating available worker replicas: %s.", err.Error()))
		}
	}

	if isJobScheduled, _ := r.isJobAlreadyScheduled(ctx, launcherName, hjob.Namespace); isJobScheduled {
		log.Info("Skip patching: job is already scheduled.")
		return ctrl.Result{}, nil
	}

	ipAddresses, err := r.getWorkersIPAddresses(ctx, workers)
	launcher, err := r.desiredLauncherJob(hjob, launcherName, ipAddresses)
	if err != nil {
		return ctrl.Result{}, err
	}

	err = r.Patch(ctx, &launcher, client.Apply, applyOpts...)
	if err != nil {
		log.Info(fmt.Sprintf("Error in patching launcher: %s.", err.Error()))
		return ctrl.Result{Requeue: true}, nil
	}

	return ctrl.Result{}, nil
}

func (r *HorovodJobReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&aiv1alpha1.HorovodJob{}).
		Owns(&appsv1.Deployment{}).
		Owns(&batchv1.Job{}).
		Complete(r)
}

func (r *HorovodJobReconciler) desiredWorkers(hjob aiv1alpha1.HorovodJob, workersName string) (appsv1.Deployment, error) {
	defaultMode := int32(0600)
	deployment := appsv1.Deployment{
		TypeMeta: metav1.TypeMeta{APIVersion: appsv1.SchemeGroupVersion.String(), Kind: "Deployment"},
		ObjectMeta: metav1.ObjectMeta{
			Name:      workersName,
			Namespace: hjob.Namespace,
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: hjob.Spec.WorkersSpec.Replicas,
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{"HorovodJob": hjob.Name, "role": "worker"},
			},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"HorovodJob": hjob.Name, "role": "worker"},
				},
				Spec: corev1.PodSpec{
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
					},
					Containers: []corev1.Container{
						{
							Name:  "worker",
							Image: hjob.Spec.WorkersSpec.Image,
							VolumeMounts: []corev1.VolumeMount{
								{
									Name:      "sshkeys",
									MountPath: "/etc/secrets",
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

	if err := ctrl.SetControllerReference(&hjob, &deployment, r.Scheme); err != nil {
		return deployment, err
	}

	return deployment, nil
}

func (r *HorovodJobReconciler) areWorkersReady(ctx context.Context, name, namespace string) (bool, error) {
	var workers appsv1.Deployment
	key := types.NamespacedName{
		Namespace: namespace,
		Name:      name,
	}

	if err := r.Get(ctx, key, &workers); err != nil {
		return false, err
	}

	if workers.Status.AvailableReplicas == *workers.Spec.Replicas && workers.Status.UpdatedReplicas == *workers.Spec.Replicas {
		return true, nil
	}

	return false, nil
}

func (r *HorovodJobReconciler) getWorkersIPAddresses(ctx context.Context, depl appsv1.Deployment) ([]string, error) {
	var podlist corev1.PodList
	selector, _ := metav1.LabelSelectorAsSelector(depl.Spec.Selector)

	listOptions := []client.ListOption{
		client.MatchingLabelsSelector{Selector: selector},
		client.InNamespace(depl.Namespace),
	}

	if err := r.List(ctx, &podlist, listOptions...); err != nil {
		return nil, err
	}

	res := make([]string, len(podlist.Items))
	for i, pod := range podlist.Items {
		res[i] = pod.Status.PodIP
	}

	return res, nil
}

func (r *HorovodJobReconciler) desiredLauncherJob(hjob aiv1alpha1.HorovodJob, launcherName string, workersIP []string) (batchv1.Job, error) {
	defaultMode := int32(0600)
	one := int32(1)
	sshSetupCommand := "mkdir -p /root/.ssh; cp /etc/secrets/* /root/.ssh/; chmod 644 /root/.ssh/authorized_keys;"
	horovodArgs := fmt.Sprintf("-np %d -H %s", len(workersIP), strings.Join(workersIP, ":1,")+":1")
	pythonCommand := strings.Join(hjob.Spec.LauncherSpec.PythonCommand, " ")
	wholeCommand := fmt.Sprintf("%s horovodrun --gloo --network-interface eth0 -p 12345 %s %s", sshSetupCommand, horovodArgs, pythonCommand)

	job := batchv1.Job{
		TypeMeta: metav1.TypeMeta{APIVersion: batchv1.SchemeGroupVersion.String(), Kind: "Job"},
		ObjectMeta: metav1.ObjectMeta{
			Name:      launcherName,
			Namespace: hjob.Namespace,
		},
		Spec: batchv1.JobSpec{
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{"HorovodJob": hjob.Name, "role": "launcher"},
			},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"HorovodJob": hjob.Name, "role": "launcher"},
				},
				Spec: corev1.PodSpec{
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
					},
					Containers: []corev1.Container{
						{
							Name:  "launcher",
							Image: hjob.Spec.LauncherSpec.Image,
							VolumeMounts: []corev1.VolumeMount{
								{
									Name:      "sshkeys",
									MountPath: "/etc/secrets",
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
				},
			},
			BackoffLimit: &one,
		},
	}

	if err := ctrl.SetControllerReference(&hjob, &job, r.Scheme); err != nil {
		return job, err
	}

	return job, nil
}

func (r *HorovodJobReconciler) isJobAlreadyScheduled(ctx context.Context, name, namespace string) (bool, error) {
	var job batchv1.Job
	key := types.NamespacedName{
		Namespace: namespace,
		Name:      name,
	}

	if err := r.Get(ctx, key, &job); err != nil {
		return false, err
	}

	return true, nil
}
