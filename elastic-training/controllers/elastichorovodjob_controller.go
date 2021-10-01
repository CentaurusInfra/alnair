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

	"github.com/go-logr/logr"
	"github.com/prometheus/common/log"
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

	sigsv1alpha1 "sigs.k8s.io/scheduler-plugins/pkg/apis/scheduling/v1alpha1"
	aiv1alpha1 "elastictraining/api/v1alpha1"
)

// ElasticHorovodJobReconciler reconciles a ElasticHorovodJob object
type ElasticHorovodJobReconciler struct {
	client.Client
	Log    logr.Logger
	Scheme *runtime.Scheme
}

// +kubebuilder:rbac:groups=ai.centauruscloud.io,resources=elastichorovodjobs,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=ai.centauruscloud.io,resources=elastichorovodjobs/status,verbs=get;update;patch

func (r *ElasticHorovodJobReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := r.Log.WithValues("ElasticHorovodJob", req.NamespacedName)
	log.Info("Start reconciling")

	var ehjob aiv1alpha1.ElasticHorovodJob
	if err := r.Get(ctx, req.NamespacedName, &ehjob); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	serviceName := fmt.Sprintf("elastichorovodjob-%s", ehjob.Name)
	workersName := fmt.Sprintf("elastichorovodjob-%s-workers", ehjob.Name)
	launcherName := fmt.Sprintf("elastichorovodjob-%s-launcher", ehjob.Name)
	pgName := fmt.Sprintf("podgroup-ehjob-%s", ehjob.Name)

	if ehjob.Status.Workers != "statefulset.apps/"+workersName ||
		ehjob.Status.Launcher != "job.batch/"+launcherName {
		ehjob.Status.Workers = "statefulset.apps/" + workersName
		ehjob.Status.Launcher = "job.batch/" + launcherName

		if err := r.Status().Update(ctx, &ehjob); err != nil {
			log.Info(fmt.Sprintf("Error in updating launcher and workers names: %s.", err.Error()))
		}
	}

	applyOpts := []client.PatchOption{client.ForceOwnership, client.FieldOwner("elastichorovodjob-controller")}

	if isJobScheduled, job, _ := r.isJobAlreadyScheduled(ctx, launcherName, ehjob.Namespace); isJobScheduled {
		if isJobCompleteOrFailed(job) {
			log.Info("Job has either completed or errored, releasing GPU resources.")
			if err := r.releaseResources(ctx, serviceName, workersName, ehjob.Namespace); err != nil {
				if !errors.IsNotFound(err) {
					log.Info(fmt.Sprintf("Error in cleaning up resources: %s.", err.Error()))
				}
			}
			return ctrl.Result{}, nil
		}

		//check if targetreplicas has been changed while job runs
		if isJobRunning(job) {
			workers, err := r.getWorkers(ctx, ehjob.Namespace, workersName)

			if err != nil {
				if !errors.IsNotFound(err) {
					log.Info(fmt.Sprintf("Error in querying workers: %s.", err.Error()))
				}
			}

			if *ehjob.Spec.WorkersSpec.TargetReplicas != *workers.Spec.Replicas {
				log.Info(fmt.Sprintf("Scaling workers and PodGroup from current %d to target %d", *workers.Spec.Replicas, *ehjob.Spec.WorkersSpec.TargetReplicas))
				workers, err := r.desiredWorkers(ehjob, workersName, serviceName)
				if err != nil {
					return ctrl.Result{}, nil
				}

				if err := r.Patch(ctx, &workers, client.Apply, applyOpts...); err != nil {
					log.Info(fmt.Sprintf("Error in patching workers: %s.", err.Error()))
					return ctrl.Result{Requeue: true}, nil
				}

				log.Info("Successfully scaled workers.")

				if r.createAndPatchPodGroup(ctx, ehjob, pgName, applyOpts) {
					log.Info("Successfully scaled PodGroup.")
				} else {
					return ctrl.Result{RequeueAfter: 5}, nil
				}
			}
		}

		log.Info("Skip patching: job is already scheduled.")
		return ctrl.Result{}, nil
	}

	if !r.serviceExists(ctx, serviceName, ehjob.Namespace) {
		svc, err := r.desiredService(ehjob, serviceName)
		if err != nil {
			return ctrl.Result{}, err
		}

		err = r.Patch(ctx, &svc, client.Apply, applyOpts...)
		if err != nil {
			log.Info(fmt.Sprintf("Error in patching service: %s.", err.Error()))
			return ctrl.Result{Requeue: true}, nil
		}
	}

	if ehjob.Spec.WorkersSpec.TargetReplicas == nil {
		log.Info("workers target replicas have not been set by scheduler yet.")
		return ctrl.Result{}, nil
	}

	//worker replicas are set correctly, but pod state is stuck in pending
	//scale down by only one, as this only occurs during the race condition

	if workers, err := r.getWorkers(ctx, ehjob.Namespace, workersName); err != nil {
		if !errors.IsNotFound(err) {
			log.Info(fmt.Sprintf("Error in querying workers: %s.", err.Error()))
		}
	} else {
		if workers.Status.ReadyReplicas == 0 && *workers.Spec.Replicas == *ehjob.Spec.WorkersSpec.TargetReplicas {
			time.Sleep(15 * time.Second)

			//re-get workers
			if workers, err = r.getWorkers(ctx, ehjob.Namespace, workersName); err != nil {
				if !errors.IsNotFound(err) {
					log.Info(fmt.Sprintf("Error in querying workers: %s.", err.Error()))
				}
			}

			if workers.Status.ReadyReplicas == 0 {
				(*ehjob.Spec.WorkersSpec.TargetReplicas) -= 1
				log.Info(fmt.Sprintf("Workers currently all pending, changing target replicas to %d", *ehjob.Spec.WorkersSpec.TargetReplicas))
				if err := r.Update(ctx, &ehjob); err != nil {
					log.Info(fmt.Sprintf("Error in updating target replicas for ElasticHorovodJob %s/%s: %s.",
						ehjob.Namespace, ehjob.Name, err.Error()))
					return ctrl.Result{Requeue: true}, nil
				}

				//workers scaled down by one, now scale the workers and the podgroup
				log.Info("Deleting StatefulSet to be recreated")
				if err := r.deleteWorkers(ctx, workersName, ehjob.Namespace); err != nil {
					log.Info("Error in deleting StatefulSet")
				}
				time.Sleep(5 * time.Second)
				return ctrl.Result{Requeue: true}, nil
			}

		}
	}

	workersReady, err := r.areWorkersReady(ctx, workersName, ehjob)

	if err != nil {
		if !errors.IsNotFound(err) {
			log.Info(fmt.Sprintf("Error in querying workers: %s.", err.Error()))
		}
	}

	if !workersReady {
		workers, err := r.desiredWorkers(ehjob, workersName, serviceName)
		if err != nil {
			return ctrl.Result{}, err
		}

		err = r.Patch(ctx, &workers, client.Apply, applyOpts...)
		if err != nil {
			log.Info(fmt.Sprintf("Error in patching workers: %s.", err.Error()))
			return ctrl.Result{Requeue: true}, nil
		}

		if r.createAndPatchPodGroup(ctx, ehjob, pgName, applyOpts) {
			log.Info(fmt.Sprintf("Patched PodGroup %s", pgName))
		} else {
			return ctrl.Result{RequeueAfter: 5}, nil
		}

		if isReady, err := r.areWorkersReady(ctx, workers.Name, ehjob); err != nil {
			log.Info(fmt.Sprintf("Error in workers: %s.", err.Error()))
			return ctrl.Result{Requeue: true}, nil
		} else {
			if !isReady {
				log.Info("Workers are not ready yet.")
				return ctrl.Result{}, nil
			}
		}
	}

	// Now workers are ready
	if ehjob.Status.AvailableWorkerReplicas != *ehjob.Spec.WorkersSpec.TargetReplicas {
		ehjob.Status.AvailableWorkerReplicas = *ehjob.Spec.WorkersSpec.TargetReplicas
		if err := r.Status().Update(ctx, &ehjob); err != nil {
			log.Info(fmt.Sprintf("Error in updating available worker replicas: %s.", err.Error()))
		}
	}

	launcher, err := r.desiredLauncherJob(ehjob, launcherName, serviceName)
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

func (r *ElasticHorovodJobReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&aiv1alpha1.ElasticHorovodJob{}).
		Owns(&corev1.Service{}).
		Owns(&appsv1.StatefulSet{}).
		Owns(&batchv1.Job{}).
		Complete(r)
}

func (r *ElasticHorovodJobReconciler) desiredService(ehjob aiv1alpha1.ElasticHorovodJob, svcName string) (corev1.Service, error) {
	svc := corev1.Service{
		TypeMeta: metav1.TypeMeta{APIVersion: corev1.SchemeGroupVersion.String(), Kind: "Service"},
		ObjectMeta: metav1.ObjectMeta{
			Name:      svcName,
			Namespace: ehjob.Namespace,
		},
		Spec: corev1.ServiceSpec{
			ClusterIP: "None",
			Selector:  map[string]string{"ElasticHorovodJob": ehjob.Name, "role": "worker"},
		},
	}

	if err := ctrl.SetControllerReference(&ehjob, &svc, r.Scheme); err != nil {
		return svc, err
	}

	return svc, nil
}

// desiredWorkers returns a StatefulSet. StatefulSet is slow but Deployment could crash the ealstic horovod training script
// in the case when scaling down many workers at a time.
// Deployment used to enable coscheduling
func (r *ElasticHorovodJobReconciler) desiredWorkers(ehjob aiv1alpha1.ElasticHorovodJob, workersName, svcName string) (appsv1.StatefulSet, error) {
	defaultMode := int32(0600)
	statefulset := appsv1.StatefulSet{
		TypeMeta: metav1.TypeMeta{APIVersion: appsv1.SchemeGroupVersion.String(), Kind: "StatefulSet"},
		ObjectMeta: metav1.ObjectMeta{
			Name:      workersName,
			Namespace: ehjob.Namespace,
		},
		Spec: appsv1.StatefulSetSpec{
			ServiceName:         svcName,
			Replicas:            ehjob.Spec.WorkersSpec.TargetReplicas,
			PodManagementPolicy: appsv1.PodManagementPolicyType(string(appsv1.ParallelPodManagement)),
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{"ElasticHorovodJob": ehjob.Name, "role": "worker"},
			},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{ //add labels such as podgroup # and statefulset
					Labels: map[string]string{"ElasticHorovodJob": ehjob.Name, "role": "worker",
						"pod-group.scheduling.sigs.k8s.io": fmt.Sprintf("podgroup-ehjob-%s", ehjob.Name),
					},
				},
				Spec: corev1.PodSpec{
					//SchedulerName: "default-scheduler", 
					SchedulerName: "coscheduling-only",
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
							Image: ehjob.Spec.WorkersSpec.Image,
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

	if err := ctrl.SetControllerReference(&ehjob, &statefulset, r.Scheme); err != nil {
		return statefulset, err
	}

	return statefulset, nil
}

func (r *ElasticHorovodJobReconciler) serviceExists(ctx context.Context, name, namespace string) bool {
	var svc corev1.Service
	key := types.NamespacedName{
		Namespace: namespace,
		Name:      name,
	}

	if err := r.Get(ctx, key, &svc); err != nil {
		return false
	}

	return true
}

func (r *ElasticHorovodJobReconciler) areWorkersReady(ctx context.Context, name string, ehjob aiv1alpha1.ElasticHorovodJob) (bool, error) {
	workers, err := r.getWorkers(ctx, ehjob.Namespace, name)
	if err != nil {
		return false, err
	}

	targetReplicas := *ehjob.Spec.WorkersSpec.TargetReplicas
	if workers.Status.ReadyReplicas == targetReplicas && workers.Status.UpdatedReplicas == targetReplicas {
		return true, nil
	}

	return false, nil
}

func (r *ElasticHorovodJobReconciler) desiredLauncherJob(ehjob aiv1alpha1.ElasticHorovodJob, launcherName, svcName string) (batchv1.Job, error) {
	sshkeysMode := int32(0600)
	scriptMode := int32(0744)
	one := int32(1)
	sshSetupCommand := "mkdir -p /root/.ssh; cp /etc/secrets/* /root/.ssh/; chmod 644 /root/.ssh/authorized_keys;"
	scriptSetupCmd := fmt.Sprintf("mkdir -p /scripts; mkdir -p /elastic_scripts; cp /etc/scripts/* /scripts/; sed -i 's/SERVICENAME/%s/' /scripts/discover_hosts.sh;", svcName)
	horovodArgs := fmt.Sprintf("-np %d --max-np %d --host-discovery-script /scripts/discover_hosts.sh",
		*ehjob.Spec.WorkersSpec.MinReplicas,
		*ehjob.Spec.WorkersSpec.MaxReplicas)
	pythonCommand := strings.Join(ehjob.Spec.LauncherSpec.PythonCommand, " ")
	//sleepCommand := "sleep 600;"
	sleepCommand := ""
	wholeCommand := fmt.Sprintf("%s %s %s horovodrun -p 12345 %s %s", sshSetupCommand, scriptSetupCmd, sleepCommand, horovodArgs, pythonCommand)

	job := batchv1.Job{
		TypeMeta: metav1.TypeMeta{APIVersion: batchv1.SchemeGroupVersion.String(), Kind: "Job"},
		ObjectMeta: metav1.ObjectMeta{
			Name:      launcherName,
			Namespace: ehjob.Namespace,
		},
		Spec: batchv1.JobSpec{
			Template: corev1.PodTemplateSpec{
				//commented because podgroup only supports single-object podgroup (podgroup
				//label is enabled in worker statefulset)

				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"pod-group.scheduling.sigs.k8s.io": fmt.Sprintf("podgroup-ehjob-%s", ehjob.Name),
					},
				},
				Spec: corev1.PodSpec{
					SchedulerName: "coscheduling-only", //coscheduling
					NodeSelector: map[string]string{
						"kubernetes.io/hostname": "titan34",
					},
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
							Image: ehjob.Spec.LauncherSpec.Image,
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
			//Selector: &metav1.LabelSelector{
			//	MatchLabels: map[string]string{"kubernetes.io/hostname": "titan34"},
			//},
		},
	}

	if err := ctrl.SetControllerReference(&ehjob, &job, r.Scheme); err != nil {
		return job, err
	}

	return job, nil
}

func (r *ElasticHorovodJobReconciler) isJobAlreadyScheduled(ctx context.Context, name, namespace string) (bool, batchv1.Job, error) {
	var job batchv1.Job
	key := types.NamespacedName{
		Namespace: namespace,
		Name:      name,
	}

	if err := r.Get(ctx, key, &job); err != nil {
		return false, job, err
	}

	return true, job, nil
}

func (r *ElasticHorovodJobReconciler) deleteService(ctx context.Context, name, namespace string) error {
	var svc corev1.Service
	key := types.NamespacedName{
		Namespace: namespace,
		Name:      name,
	}

	if err := r.Get(ctx, key, &svc); err != nil {
		return nil
	}

	zero := int64(0)
	deletepolicy := metav1.DeletePropagationForeground
	deleteOpts := []client.DeleteOption{&client.DeleteOptions{
		GracePeriodSeconds: &zero,
		PropagationPolicy:  &deletepolicy,
	}}
	return r.Delete(ctx, &svc, deleteOpts...)
}

func (r *ElasticHorovodJobReconciler) deleteWorkers(ctx context.Context, name, namespace string) error {
	var workers appsv1.StatefulSet
	key := types.NamespacedName{
		Namespace: namespace,
		Name:      name,
	}

	if err := r.Get(ctx, key, &workers); err != nil {
		return nil
	}

	zero := int64(0)
	deletepolicy := metav1.DeletePropagationForeground
	deleteOpts := []client.DeleteOption{&client.DeleteOptions{
		GracePeriodSeconds: &zero,
		PropagationPolicy:  &deletepolicy,
	}}
	return r.Delete(ctx, &workers, deleteOpts...)
}

func (r *ElasticHorovodJobReconciler) getWorkers(ctx context.Context, namespace string, workerName string) (appsv1.StatefulSet, error) {
	var workers appsv1.StatefulSet
	key := types.NamespacedName{
		Namespace: namespace,
		Name:      workerName,
	}

	if err := r.Get(ctx, key, &workers); err != nil {
		return workers, err
	}

	return workers, nil
}

func (r *ElasticHorovodJobReconciler) releaseResources(ctx context.Context, serviceName, workersName, namespace string) error {
	if err := r.deleteService(ctx, serviceName, namespace); err != nil {
		return err
	}

	if err := r.deleteWorkers(ctx, workersName, namespace); err != nil {
		return err
	}

	return nil
}

func isJobCompleteOrFailed(job batchv1.Job) bool {
	for _, c := range job.Status.Conditions {
		if c.Type == batchv1.JobComplete && c.Status == corev1.ConditionTrue {
			return true
		}

		if c.Type == batchv1.JobFailed && c.Status == corev1.ConditionTrue {
			return true
		}
	}
	return false
}
func isJobRunning(job batchv1.Job) bool {
	return job.Status.Active > 0
}

func newHostPathType(pathType string) *corev1.HostPathType {
	hostPathType := new(corev1.HostPathType)
	*hostPathType = corev1.HostPathType(pathType)
	return hostPathType
}

func (r *ElasticHorovodJobReconciler) podGroupExists(ctx context.Context, pgName string, namespace string) bool {
	var pg sigsv1alpha1.PodGroup
	key := types.NamespacedName{
		Namespace: namespace,
		Name:      pgName,
	}

	if err := r.Get(ctx, key, &pg); err != nil {
		return false
	}

	return true
}

func (r *ElasticHorovodJobReconciler) createAndPatchPodGroup(ctx context.Context, ehjob aiv1alpha1.ElasticHorovodJob, pgName string, applyOpts []client.PatchOption) bool {

	pg, err := r.createPodGroup(ctx, ehjob, pgName)
	if err != nil {
		log.Info(fmt.Sprintf("Unable to create PodGroup: %s", err.Error()))
		return false
	}

	if err = r.Patch(ctx, &pg, client.Apply, applyOpts...); err != nil {
		log.Info(fmt.Sprintf("Error in patching PodGroup: %s.", err.Error()))
		return false
	}

	if !r.podGroupExists(ctx, pg.Name, pg.Namespace) {
		log.Info("PodGroup unavailable.")
		return false
	}

	return true
}

func (r *ElasticHorovodJobReconciler) createPodGroup(ctx context.Context, ehjob aiv1alpha1.ElasticHorovodJob, pgName string) (sigsv1alpha1.PodGroup, error) {
	pg := sigsv1alpha1.PodGroup{
		TypeMeta: metav1.TypeMeta{APIVersion: "scheduling.sigs.k8s.io/v1alpha1", Kind: "PodGroup"},
		ObjectMeta: metav1.ObjectMeta{
			Name:      pgName,
			Namespace: ehjob.Namespace,
		},
		Spec: sigsv1alpha1.PodGroupSpec{
			MinMember: *ehjob.Spec.WorkersSpec.TargetReplicas, //*ehjob.Spec.WorkersSpec.MinReplicas
			MinResources: &corev1.ResourceList{
				corev1.ResourceName("nvidia.com/gpu"): *resource.NewQuantity(int64(*ehjob.Spec.WorkersSpec.TargetReplicas), resource.DecimalSI),
			},
		},
		Status: sigsv1alpha1.PodGroupStatus{
			ScheduleStartTime: metav1.Now(),
		},
	}

	if err := ctrl.SetControllerReference(&ehjob, &pg, r.Scheme); err != nil {
		return pg, err
	}

	return pg, nil
}
