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
	"reflect"

	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/CentaurusInfra/alnair/tree/main/storage-caching/k-v-store/alnairpod-operator/api/v1alpha1"
	"github.com/go-logr/logr"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const (
	APIVersion    = "core/v1"
	PodKind       = "Pod"
	SecretKind    = "Secret"
	ConfigMapKind = "ConfigMap"
	ClientImage   = "centaurusinfra/alnairpod:client"
)

// AlnairPodReconciler reconciles a AlnairPod object
type AlnairPodReconciler struct {
	client.Client
	Log    logr.Logger
	Scheme *runtime.Scheme
}

//+kubebuilder:rbac:groups=alnair.com,resources=alnairpods,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=alnair.com,resources=alnairpods/status,verbs=get;update;patch
//+kubebuilder:rbac:groups=alnair.com,resources=alnairpods/finalizers,verbs=update

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.
// TODO(user): Modify the Reconcile function to compare the state specified by
// the AlnairPod object against the actual cluster state, and then
// perform operations to make the cluster state reflect the state specified by
// the user.
//
// For more details, check Reconcile and its Result here:
// - https://pkg.go.dev/sigs.k8s.io/controller-runtime@v0.12.1/pkg/reconcile
func (r *AlnairPodReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := r.Log.WithValues("alnairpod", req.NamespacedName)
	log.Info("start reconciling")

	var alnairpod v1alpha1.AlnairPod
	err := r.Get(ctx, req.NamespacedName, &alnairpod)
	if err != nil {
		if k8serrors.IsNotFound(err) {
			// Request object not found, could have been deleted after reconcile request.
			// Owned objects are automatically garbage collected. For additional cleanup logic use finalizers.
			// Return and don't requeue
			return ctrl.Result{}, nil
		}
		// Error reading the object - requeue the request.
		return ctrl.Result{}, err
	}

	if alnairpod.DeletionTimestamp != nil {
		return ctrl.Result{}, err
	}

	// create associated resources

	// create configmap
	configmap := &corev1.ConfigMap{}
	if err := r.Get(ctx, req.NamespacedName, configmap); err != nil && k8serrors.IsNotFound(err) {
		configmap, err := r.createConfigMap(ctx, alnairpod)
		if err != nil {
			log.Error(err, fmt.Sprintf("error in creating configmap %s: %s.", configmap.Name, err.Error()))
			return ctrl.Result{}, err
		} else {
			if err := r.Create(ctx, &configmap, &client.CreateOptions{}); err != nil {
				return ctrl.Result{}, err
			}
		}
	}

	// create pod
	pod := &corev1.Pod{}
	if err := r.Get(ctx, req.NamespacedName, pod); err != nil && k8serrors.IsNotFound(err) {
		pod, err := r.createPod(ctx, alnairpod)
		if err != nil {
			log.Error(err, fmt.Sprintf("error in creating pod %s: %s.", pod.Name, err.Error()))
			return ctrl.Result{}, err
		} else {
			if err := r.Create(ctx, &pod, &client.CreateOptions{}); err != nil {
				return ctrl.Result{}, err
			}
			// associate Annotations
			data, _ := json.Marshal(alnairpod.Spec)
			if alnairpod.Annotations != nil {
				alnairpod.Annotations["spec"] = string(data)
			} else {
				alnairpod.Annotations = map[string]string{"spec": string(data)}
			}
			if err := r.Update(ctx, &alnairpod, &client.UpdateOptions{}); err != nil {
				return ctrl.Result{}, nil
			}
			return ctrl.Result{}, nil
		}
	}

	// update associated resources
	oldspec := v1alpha1.AlnairPodSpec{}
	if err := json.Unmarshal([]byte(alnairpod.Annotations["spec"]), &oldspec); err != nil {
		return ctrl.Result{}, nil
	}
	if !reflect.DeepEqual(alnairpod.Spec, oldspec) {
		// update configmap
		newconfigmap, _ := r.createConfigMap(ctx, alnairpod)
		if err := r.Update(ctx, &newconfigmap, &client.UpdateOptions{}); err != nil {
			log.Error(err, fmt.Sprintf("error in updating configmap %s", req.Name))
			return ctrl.Result{}, nil
		}

		oldpod := corev1.Pod{}
		if err := r.Get(ctx, req.NamespacedName, &oldpod); err != nil {
			log.Error(err, fmt.Sprintf("error in getting pod %s", req.Name))
			return ctrl.Result{}, err
		}
		// delete the old pod
		if err := r.Delete(ctx, &oldpod, &client.DeleteOptions{}); err != nil {
			log.Error(err, fmt.Sprintf("failed to delete old pod %s", alnairpod.Name))
			return ctrl.Result{}, err
		}

		// create a new pod
		newpod, _ := r.createPod(ctx, alnairpod)
		if err := r.Create(ctx, &newpod, &client.CreateOptions{}); err != nil {
			log.Error(err, fmt.Sprintf("error in creating pod %s: %s.", pod.Name, err.Error()))
			return ctrl.Result{}, err
		}
		return ctrl.Result{}, nil
	}
	return ctrl.Result{}, nil
}

func (r *AlnairPodReconciler) createPod(ctx context.Context, alnairpod v1alpha1.AlnairPod) (corev1.Pod, error) {
	pod := corev1.Pod{}
	spec := alnairpod.Spec
	volumes := []corev1.Volume{
		{
			Name:         "secret",
			VolumeSource: corev1.VolumeSource{Secret: &corev1.SecretVolumeSource{SecretName: spec.Secret.Name}},
		},
		{
			Name:         "jobs",
			VolumeSource: corev1.VolumeSource{ConfigMap: &corev1.ConfigMapVolumeSource{LocalObjectReference: corev1.LocalObjectReference{Name: alnairpod.Name}}},
		},
		{
			Name:         "share",
			VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}},
		},
		{
			Name:         "shmem",
			VolumeSource: corev1.VolumeSource{HostPath: &corev1.HostPathVolumeSource{Path: "/dev/shm"}},
		},
	}
	volumes = append(volumes, spec.Volumes...)
	vol_mounts := []corev1.VolumeMount{
		{Name: "secret", MountPath: "/secret"},
		{Name: "jobs", MountPath: "/jobs"},
		{Name: "share", MountPath: "/share"},
		{Name: "shmem", MountPath: "/dev/shm"},
	}

	var containers []corev1.Container
	for _, job := range spec.Jobs {
		// initialize job container
		env := job.Env
		env = append(env, corev1.EnvVar{Name: "JOBNAME", Value: job.Name})
		container := corev1.Container{
			Name:            job.Name,
			Image:           job.Image,
			ImagePullPolicy: job.ImagePullPolicy,
			WorkingDir:      job.WorkingDir,
			Env:             env,
			EnvFrom:         job.EnvFrom,
			Command:         job.Command,
			VolumeMounts:    vol_mounts,
			Ports:           job.Ports,
			Lifecycle:       job.Lifecycle,
			Resources:       job.Resources,
			TTY:             job.TTY,
			Stdin:           job.Stdin,
		}
		containers = append(containers, container)
	}

	// init the client container
	container := corev1.Container{
		Name:            "client",
		Image:           ClientImage,
		ImagePullPolicy: corev1.PullAlways,
		WorkingDir:      "/app",
		// Command:         []string{"python3", "client.py"},
		VolumeMounts: vol_mounts,
		TTY:          true,
		Stdin:        true,
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
			// Annotations: map[string]string{"k8s.v1.cni.cncf.io/networks": "macvlan-conf"},
		},
		Spec: corev1.PodSpec{
			Volumes:       volumes,
			Containers:    containers,
			RestartPolicy: corev1.RestartPolicyNever,
			NodeSelector:  spec.NodeSelector,
			HostNetwork:   spec.HostNetwork,
		},
	}

	if err := ctrl.SetControllerReference(&alnairpod, &pod, r.Scheme); err != nil {
		return pod, err
	}
	return pod, nil
}

func (r *AlnairPodReconciler) createConfigMap(ctx context.Context, alnairpod v1alpha1.AlnairPod) (corev1.ConfigMap, error) {
	configmap := corev1.ConfigMap{
		TypeMeta:   metav1.TypeMeta{Kind: ConfigMapKind, APIVersion: APIVersion},
		ObjectMeta: metav1.ObjectMeta{Name: alnairpod.Name, Namespace: alnairpod.Namespace},
		Data:       map[string]string{},
	}
	spec := alnairpod.Spec
	for _, job := range spec.Jobs {
		jobinfo := map[string]interface{}{
			"name":       job.Name,
			"datasource": job.DataSource,
		}
		if job.ConfigurationsFromConfigMap.Name != "" {
			var qos_config corev1.ConfigMap
			if err := r.Get(ctx, types.NamespacedName{Name: job.ConfigurationsFromConfigMap.Name, Namespace: alnairpod.Namespace}, &qos_config); err != nil {
				return configmap, err
			}
			jobinfo["qos"] = qos_config.Data
		} else {
			qos_data := make(map[string]interface{})
			v := reflect.ValueOf(job.Configurations)
			t := v.Type()
			for i := 0; i < v.NumField(); i++ {
				qos_data[t.Field(i).Name] = v.Field(i).Interface()
			}
			jobinfo["qos"] = qos_data
		}
		byte_arr, _ := json.Marshal(jobinfo)
		configmap.Data[fmt.Sprintf("%s.json", job.Name)] = string(byte_arr)
	}

	if err := ctrl.SetControllerReference(&alnairpod, &configmap, r.Scheme); err != nil {
		return configmap, err
	}
	return configmap, nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *AlnairPodReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&v1alpha1.AlnairPod{}).
		Owns(&corev1.ConfigMap{}).
		Owns(&corev1.Pod{}).
		Complete(r)
}
