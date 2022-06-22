/*
Copyright 2022.

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
	"encoding/json"
	"fmt"
	"io/ioutil"
	"reflect"

	"errors"

	"github.com/CentaurusInfra/alnair/tree/main/storage-caching/k-v-store/alnairpod-operator/api/v1alpha1"
	"github.com/go-logr/logr"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"

	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
)

const (
	APIVersion    = "core/v1"
	PodKind       = "Pod"
	SecretKind    = "Secret"
	ConfigMapKind = "ConfigMap"
	ClientImage   = "zhunagweikang/alnairpod:client"
)

// AlnairPodReconciler reconciles a AlnairPod object
type AlnairPodReconciler struct {
	client.Client
	Log    logr.Logger
	Scheme *runtime.Scheme
}

func (r *AlnairPodReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&v1alpha1.AlnairPod{}).
		Owns(&corev1.Pod{}).
		Complete(r)
}

//+kubebuilder:rbac:groups=alnair.com.my.domain,resources=alnairpods,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=alnair.com.my.domain,resources=alnairpods/status,verbs=get;update;patch
//+kubebuilder:rbac:groups=alnair.com.my.domain,resources=alnairpods/finalizers,verbs=update
func (r *AlnairPodReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := r.Log.WithValues("alnairpod", req.NamespacedName)
	log.Info("start reconciling")

	var alnairpod v1alpha1.AlnairPod
	if err := r.Get(ctx, req.NamespacedName, &alnairpod); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	if err := r.setDefaults(ctx, &alnairpod); err != nil {
		return reconcile.Result{}, err
	}

	var jobsName []string
	for _, job := range alnairpod.Spec.Jobs {
		jobsName = append(jobsName, fmt.Sprintf("job-%s", job.Name))
	}

	// reconcoile containers name
	if alnairpod.Status.Client != "client" || reflect.DeepEqual(alnairpod.Status.CreatedJobs, jobsName) {
		if err := r.Status().Update(ctx, &alnairpod); err != nil {
			log.Info(fmt.Sprintf("error in updating AlnairPod client and job names: %s.", err.Error()))
		}
	}

	alnairpodReady, err := r.isAlnairPodReady(ctx, alnairpod.Name, alnairpod.Namespace)
	if err != nil {
		if !k8serrors.IsNotFound(err) {
			log.Info(fmt.Sprintf("error in querying AlnairPod Pod: %s.", err.Error()))
		}
	}

	var alnairpodPod corev1.Pod
	applyOpts := []client.PatchOption{client.ForceOwnership, client.FieldOwner("alnairpod-controller")}
	if !alnairpodReady {
		alnairpodPod, err = r.desiredAlnairPod(ctx, alnairpod)
		if err != nil {
			return ctrl.Result{}, err
		}

		err = r.Patch(ctx, &alnairpodPod, client.Apply, applyOpts...)
		if err != nil {
			log.Error(err, fmt.Sprintf("error in patching alnairpod: %s.", err.Error()))
			return ctrl.Result{Requeue: true}, nil
		}

		if isReady, err := r.isAlnairPodReady(ctx, alnairpod.Name, alnairpod.Namespace); err != nil {
			log.Error(err, fmt.Sprintf("error in alnairpod: %s.", err.Error()))
			return ctrl.Result{Requeue: true}, nil
		} else {
			if !isReady {
				log.Info("alnairpod is not ready yet.")
				return ctrl.Result{}, nil
			}
		}
	}
	return ctrl.Result{}, nil
}

func (r *AlnairPodReconciler) desiredAlnairPod(ctx context.Context, alnairpod v1alpha1.AlnairPod) (corev1.Pod, error) {
	pod := corev1.Pod{}

	volumes := []corev1.Volume{
		{
			Name:         "client-secret",
			VolumeSource: corev1.VolumeSource{Secret: &corev1.SecretVolumeSource{SecretName: alnairpod.Spec.Secret.Name}},
		},
		{
			Name:         "job-config",
			VolumeSource: corev1.VolumeSource{ConfigMap: &corev1.ConfigMapVolumeSource{LocalObjectReference: corev1.LocalObjectReference{Name: alnairpod.Spec.Configurations.Name}}},
		},
		{
			Name:         "shared",
			VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}},
		},
	}
	for _, vol := range alnairpod.Spec.Volumes {
		volumes = append(volumes, vol)
	}

	var jobsinfo []map[string]interface{}
	var containers []corev1.Container
	for _, job := range alnairpod.Spec.Jobs {
		jinfo := map[string]interface{}{
			"name":       job.Name,
			"datasource": job.DataSource,
		}
		jobsinfo = append(jobsinfo, jinfo)

		// initialize job container
		vol_mounts := job.VolumeMounts
		vol_mounts = append(vol_mounts, corev1.VolumeMount{Name: "share", MountPath: "/share"})
		container := corev1.Container{
			Name:            fmt.Sprintf("job-%s", job.Name),
			Image:           job.Image,
			ImagePullPolicy: job.ImagePullPolicy,
			WorkingDir:      job.WorkingDir,
			Env:             job.Env,
			EnvFrom:         job.EnvFrom,
			Command:         job.Command,
			VolumeMounts:    vol_mounts,
			Ports:           job.Ports,
			Lifecycle:       job.Lifecycle,
			Resources:       job.Resources,
		}
		containers = append(containers, container)
	}
	file, err := json.MarshalIndent(jobsinfo, "", " ")
	err = ioutil.WriteFile("/share/jobs.json", file, 0644)
	if err != nil {
		return pod, nil
	}

	// init the client container
	container := corev1.Container{
		Name:            "client",
		Image:           ClientImage,
		ImagePullPolicy: corev1.PullAlways,
		WorkingDir:      "/apps",
		Command:         []string{"python3", "client.py"},
		VolumeMounts: []corev1.VolumeMount{
			{Name: "share", MountPath: "/share"},
			{Name: "client-secret", MountPath: "/secret"},
			{Name: "job-config", MountPath: "/config"},
		},
		TTY:   true,
		Stdin: true,
	}
	containers = append(containers, container)

	// create the pod
	pod = corev1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       PodKind,
			APIVersion: APIVersion,
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      alnairpod.Name,
			Namespace: alnairpod.Namespace,
		},
		Spec: corev1.PodSpec{
			Volumes:       volumes,
			Containers:    containers,
			RestartPolicy: corev1.RestartPolicyNever,
			NodeSelector:  alnairpod.Spec.NodeSelector,
			HostNetwork:   alnairpod.Spec.HostNetwork,
		},
	}

	if err := ctrl.SetControllerReference(&alnairpod, &pod, r.Scheme); err != nil {
		return pod, err
	}
	return pod, nil
}

func (r *AlnairPodReconciler) isAlnairPodReady(ctx context.Context, name, namespace string) (bool, error) {
	var alnairpod corev1.Pod
	key := types.NamespacedName{
		Namespace: namespace,
		Name:      name,
	}

	if err := r.Get(ctx, key, &alnairpod); err != nil {
		return false, err
	}

	for _, container_status := range alnairpod.Status.ContainerStatuses {
		if !container_status.Ready {
			return false, nil
		}
	}
	return true, nil
}

func (r *AlnairPodReconciler) setDefaults(ctx context.Context, alnairpod *v1alpha1.AlnairPod) (err error) {
	if len(alnairpod.Spec.Jobs) == 0 {
		return errors.New("please specify at least one job")
	}
	var job_config corev1.ConfigMap
	if alnairpod.Spec.Configurations.Name == "" {
		r.Log.Info("not found Job QoS configurations")
	} else if err := r.Get(ctx, types.NamespacedName{Namespace: alnairpod.Namespace, Name: alnairpod.Spec.Configurations.Name}, &job_config); err != nil {
		r.Log.Error(err, fmt.Sprintf("failed to load ConfigMap %s", alnairpod.Spec.Configurations.Name))
	} else {
		default_qos := reflect.ValueOf(DefaultQoS)
		fields := default_qos.Type()
		for i := 0; i < default_qos.NumField(); i++ {
			job_config.Data[fields.Field(i).Name] = default_qos.Field(i).Interface().(string)
		}
		if err := r.Update(ctx, &job_config, &client.UpdateOptions{}); err != nil {
			r.Log.Error(err, fmt.Sprintf("failed to update existing ConfigMap %s", alnairpod.Name))
			return err
		}
	}

	for i, job := range alnairpod.Spec.Jobs {
		if len(job.ImagePullPolicy) == 0 {
			r.Log.Info(fmt.Sprintf("setting default container image pull policy: %s for job %s", corev1.PullAlways, job.Name))
			alnairpod.Spec.Jobs[i].ImagePullPolicy = corev1.PullAlways
		}
		if len(job.VolumeMounts) == 0 {
			r.Log.Info("setting default volume mount")
			alnairpod.Spec.Jobs[i].VolumeMounts = []corev1.VolumeMount{}
		}
	}
	return nil
}
