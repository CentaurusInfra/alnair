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

	ClientImage        = "zhunagweikang/alnairpodclient:latest"
	SharedVolMountPath = "/data"
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

	clientName := fmt.Sprintf("alnairpod-%s-client", alnairpod.Name)
	var jobsName []string
	for _, job := range alnairpod.Spec.Jobs {
		jobsName = append(jobsName, fmt.Sprintf("alnairpod-%s-job-%s", alnairpod.Name, job.Name))
	}

	// reconcoile containers name
	if alnairpod.Status.Client != clientName || reflect.DeepEqual(alnairpod.Status.CreatedJobs, jobsName) {
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

	// client shares job registration response message with job through a shared volume
	shared_vol := corev1.Volume{
		Name:         fmt.Sprintf("alnairpod-%s", alnairpod.Name),
		VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}},
	}
	shared_vol_mnt := corev1.VolumeMount{
		Name:      fmt.Sprintf("alnairpod-%s", alnairpod.Name),
		ReadOnly:  true,
		MountPath: SharedVolMountPath,
	}
	volumes := alnairpod.Spec.Volumes
	volumes = append(volumes, shared_vol)

	// collect job information and pass to client container through env variable
	var secret corev1.Secret
	if err := r.Get(ctx, types.NamespacedName{Name: alnairpod.Spec.Secret.Name, Namespace: alnairpod.Namespace}, &secret); err != nil {
		return pod, err
	}
	alnairpodjobinfo := make(map[string]interface{})
	alnairpodjobinfo["credential"] = map[string]interface{}{
		"server_address": secret.Data["server_address"],
		"server_port":    secret.Data["server_port"],
		"username":       secret.Data["username"],
		"password":       secret.Data["password"],
	}
	var jobsinfo []map[string]interface{}
	var containers []corev1.Container
	for _, job := range alnairpod.Spec.Jobs {
		jobinfo := make(map[string]interface{})
		var s3auth_secret corev1.Secret
		if err := r.Get(ctx, types.NamespacedName{Name: job.DataSource.Secret.Name, Namespace: alnairpod.Namespace}, &s3auth_secret); err != nil {
			return pod, err
		}
		jobinfo["name"] = job.Name
		jobinfo["dataset"] = job.DataSource.Name
		jobinfo["s3auth"] = map[string]interface{}{
			"aws_access_key_id":     s3auth_secret.Data["aws_access_key_id"],
			"aws_secret_access_key": s3auth_secret.Data["aws_secret_access_key"],
			"region_name":           s3auth_secret.Data["region_name"],
			"bucket":                job.DataSource.Bucket,
			"keys":                  job.DataSource.Keys,
		}
		var qos_configmap corev1.ConfigMap
		if err := r.Get(ctx, types.NamespacedName{Name: job.Configurations.Name, Namespace: alnairpod.Namespace}, &qos_configmap); err != nil {
			return pod, err
		}
		jobinfo["QoS"] = map[string]interface{}{
			"useCache":         qos_configmap.Data["useCache"],
			"flushFreq":        qos_configmap.Data["flushFreq"],
			"durabilityInMem":  qos_configmap.Data["durabilityInMem"],
			"durabilityInDisk": qos_configmap.Data["durabilityInDisk"],
		}
		jobsinfo = append(jobsinfo, jobinfo)

		// initialize job container
		vol_mounts := job.VolumeMounts
		vol_mounts = append(vol_mounts, shared_vol_mnt)
		container := corev1.Container{
			Name:            fmt.Sprintf("alnairpod-%s-job-%s", alnairpod.Name, job.Name),
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
	alnairpodjobinfo["jobs"] = jobsinfo
	alnairpodjobinfoJsonStr, err := json.Marshal(alnairpodjobinfo)
	if err != nil {
		return pod, nil
	}

	// init the client container
	shared_vol_mnt.ReadOnly = false
	container := corev1.Container{
		Name:            fmt.Sprintf("alnairpod-%s-client", alnairpod.Name),
		Image:           ClientImage,
		ImagePullPolicy: corev1.PullAlways,
		WorkingDir:      "/apps",
		Command:         []string{"python3", "client.py"},
		Env:             []corev1.EnvVar{{Name: "ALNAIRJOBs", Value: string(alnairpodjobinfoJsonStr)}},
		VolumeMounts:    []corev1.VolumeMount{shared_vol_mnt},
		TTY:             true,
		Stdin:           true,
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

	for i, job := range alnairpod.Spec.Jobs {
		if len(job.ImagePullPolicy) == 0 {
			r.Log.Info(fmt.Sprintf("setting default container image pull policy: %s for job %s", corev1.PullAlways, job.Name))
			alnairpod.Spec.Jobs[i].ImagePullPolicy = corev1.PullAlways
		}
		if len(job.VolumeMounts) == 0 {
			r.Log.Info("setting default volume mount")
			alnairpod.Spec.Jobs[i].VolumeMounts = []corev1.VolumeMount{}
		}

		default_config_data := ToConfigmapData(DefaultQoS, false).(map[string]string)
		var job_config corev1.ConfigMap
		if err := r.Get(ctx, types.NamespacedName{Namespace: alnairpod.Namespace, Name: job.Configurations.Name}, &job_config); err != nil {
			r.Log.Info("setting default QoS configurations")
			configmap := corev1.ConfigMap{
				TypeMeta:   metav1.TypeMeta{Kind: ConfigMapKind, APIVersion: APIVersion},
				ObjectMeta: metav1.ObjectMeta{Name: job.Name},
				Data:       default_config_data,
			}
			if err := r.Create(ctx, &configmap, &client.CreateOptions{}); err != nil {
				r.Log.Error(err, fmt.Sprintf("fail to create default ConfigMap for job %s", job.Name))
				return err
			}
			alnairpod.Spec.Jobs[i].Configurations = v1alpha1.ConfigMapRef{Name: job.Name}
		} else {
			for field := range job_config.Data {
				default_config_data[field] = job_config.Data[field]
			}
			// update the existing ConfigMap
			job_config.Data = default_config_data
			if err := r.Update(ctx, &job_config, &client.UpdateOptions{}); err != nil {
				r.Log.Error(err, fmt.Sprintf("failed to update existing ConfigMap %s", job.Configurations.Name))
				return err
			}
		}
	}
	return nil
}
