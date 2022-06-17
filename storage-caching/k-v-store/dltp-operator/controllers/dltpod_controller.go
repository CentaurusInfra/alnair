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
	"fmt"
	"math/rand"
	"reflect"
	"time"

	"github.com/CentaurusInfra/alnair/tree/main/storage-caching/k-v-store/dltp-operator/api/v1alpha1"
	"github.com/CentaurusInfra/alnair/tree/main/storage-caching/k-v-store/dltp-operator/pkg/log"
	"github.com/CentaurusInfra/alnair/tree/main/storage-caching/k-v-store/dltp-operator/pkg/notifications/event"

	"github.com/pkg/errors"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"

	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
)

type reconcileError struct {
	err     error
	counter uint64
}

const (
	APIVersion    = "core/v1"
	SecretKind    = "Secret"
	ConfigMapKind = "ConfigMap"
)

var reconcileErrors = map[string]reconcileError{}
var logx = log.Log

// DLTPodReconciler reconciles a DLTPod object
type DLTPodReconciler struct {
	Client                    client.Client
	Scheme                    *runtime.Scheme
	DLTPAPIConnectionSettings dltpclient.DLTPAPIConnectionSettings
	ClientSet                 kubernetes.Clientset
	Config                    rest.Config
	NotificationEvents        *chan event.Event
	KubernetesClusterDomain   string
}

//+kubebuilder:rbac:groups=alnair.com.my.domain,resources=dltpods,verbs=get;list;watch;create;update;patch;delete
//+kubebuilder:rbac:groups=alnair.com.my.domain,resources=dltpods/status,verbs=get;update;patch
//+kubebuilder:rbac:groups=alnair.com.my.domain,resources=dltpods/finalizers,verbs=update

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.
// TODO(user): Modify the Reconcile function to compare the state specified by
// the DLTPod object against the actual cluster state, and then
// perform operations to make the cluster state reflect the state specified by
// the user.
//
// For more details, check Reconcile and its Result here:
// - https://pkg.go.dev/sigs.k8s.io/controller-runtime@v0.12.1/pkg/reconcile
func (r *DLTPodReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	reconcileFailLimit := uint64(10)
	logger := logx.WithValues("cr", req.Name)
	logger.V(log.VDebug).Info("Reconciling DLTPod")

	result, dltpod, err := r.reconcile(req)
	if err != nil && apierrors.IsConflict(err) {
		return reconcile.Result{Requeue: true}, nil
	} else if err != nil {
		lastErrors, found := reconcileErrors[req.Name]
		if found {
			if err.Error() == lastErrors.err.Error() {
				lastErrors.counter++
			} else {
				lastErrors.counter = 1
				lastErrors.err = err
			}
		} else {
			lastErrors = reconcileError{
				err:     err,
				counter: 1,
			}
		}
		reconcileErrors[req.Name] = lastErrors
		if lastErrors.counter >= reconcileFailLimit {
			if log.Debug {
				logger.V(log.VWarn).Info(fmt.Sprintf("Reconcile loop failed %d times with the same errors, giving up: %+v", reconcileFailLimit, err))
			} else {
				logger.V(log.VWarn).Info(fmt.Sprintf("Reconcile loop failed %d times with the same errors, giving up: %s", reconcileFailLimit, err))
			}

			*r.NotificationEvents <- event.Event{
				DLTPod: *dltpod,
				Phase:  event.PhaseBase,
				Level:  v1alpha1.NotificationLevelWarning,
				Reason: reason.NewReconcileLoopFailed(
					reason.OperatorSource,
					[]string{fmt.Sprintf("Reconcile loop failed %d times with the same errors, giving up: %s", reconcileFailLimit, err)},
				),
			}
			return reconcile.Result{Requeue: false}, nil
		}

		if log.Debug {
			logger.V(log.VWarn).Info(fmt.Sprintf("Reconcile loop failed: %+v", err))
		} else if err.Error() != fmt.Sprintf("Operation cannot be fulfilled on dltpod.dltpod.io \"%s\": the object has been modified; please apply your changes to the latest version and try again", request.Name) {
			logger.V(log.VWarn).Info(fmt.Sprintf("Reconcile loop failed: %s", err))
		}

		return reconcile.Result{Requeue: true}, nil
	}
	if result.Requeue && result.RequeueAfter == 0 {
		result.RequeueAfter = time.Duration(rand.Intn(10)) * time.Millisecond
	}
	return result, nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *DLTPodReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&alnaircomv1alpha1.DLTPod{}).
		Complete(r)
}

func (r *DLTPodReconciler) reconcile(request reconcile.Request) (reconcile.Result, *v1alpha1.DLTPod, error) {
	logger := logx.WithValues("cr", request.Name)
	// Fetch the DLTPod instance
	dltpod := &v1alpha1.DLTPod{}
	var err error
	err = r.Client.Get(context.TODO(), request.NamespacedName, dltpod)
	if err != nil {
		if apierrors.IsNotFound(err) {
			// Request object not found, could have been deleted after reconcile request.
			// Owned objects are automatically garbage collected. For additional cleanup logic use finalizers.
			// Return and don't requeue
			return reconcile.Result{}, nil, nil
		}
		// Error reading the object - requeue the request.
		return reconcile.Result{}, nil, errors.WithStack(err)
	}
	var requeue bool
	requeue, err = r.setDefaults(dltpod)
	if err != nil {
		return reconcile.Result{}, dltpod, err
	}
	if requeue {
		return reconcile.Result{Requeue: true}, dltpod, nil
	}

	config := r.newDLTPodReconciler(dltpod)
	// Reconcile QoS configuration
	qosConfiguration := base.New(config, r.DLTPodAPIConnectionSettings)

	var baseMessages []string
	baseMessages, err = qosConfiguration.Validate(dltpod)
	if err != nil {
		return reconcile.Result{}, dltpod, err
	}
	if len(baseMessages) > 0 {
		message := "Validation of base configuration failed, please correct DLTPod CR."
		*r.NotificationEvents <- event.Event{
			DLTPod: *dltpod,
			Phase:  event.PhaseBase,
			Level:  v1alpha1.NotificationLevelWarning,
			Reason: reason.NewBaseConfigurationFailed(reason.HumanSource, []string{message}, append([]string{message}, baseMessages...)...),
		}
		logger.V(log.VWarn).Info(message)
		for _, msg := range baseMessages {
			logger.V(log.VWarn).Info(msg)
		}
		return reconcile.Result{}, dltpod, nil // don't requeue
	}

	var result reconcile.Result
	var dltpodClient dltpodclient.DLTPod
	result, dltpodClient, err = qosConfiguration.Reconcile()
	if err != nil {
		return reconcile.Result{}, dltpod, err
	}
	if result.Requeue {
		return result, dltpod, nil
	}
	if dltpodClient == nil {
		return reconcile.Result{Requeue: false}, dltpod, nil
	}

	if dltpod.Status.BaseConfigurationCompletedTime == nil {
		now := metav1.Now()
		dltpod.Status.BaseConfigurationCompletedTime = &now
		err = r.Client.Status().Update(context.TODO(), dltpod)
		if err != nil {
			return reconcile.Result{}, dltpod, errors.WithStack(err)
		}

		message := fmt.Sprintf("Base configuration phase is complete, took %s",
			dltpod.Status.BaseConfigurationCompletedTime.Sub(dltpod.Status.ProvisionStartTime.Time))
		*r.NotificationEvents <- event.Event{
			DLTPod: *dltpod,
			Phase:  event.PhaseBase,
			Level:  v1alpha1.NotificationLevelInfo,
			Reason: reason.NewBaseConfigurationComplete(reason.OperatorSource, []string{message}),
		}
		logger.Info(message)
	}

	// Reconcile casc, seedjobs and backups
	userConfiguration := user.New(config, jenkinsClient)

	var messages []string
	messages, err = userConfiguration.Validate(dltpod)
	if err != nil {
		return reconcile.Result{}, dltpod, err
	}
	if len(messages) > 0 {
		message := "Validation of user configuration failed, please correct DLTPod CR"
		*r.NotificationEvents <- event.Event{
			DLTPod: *dltpod,
			Phase:  event.PhaseUser,
			Level:  v1alpha1.NotificationLevelWarning,
			Reason: reason.NewUserConfigurationFailed(reason.HumanSource, []string{message}, append([]string{message}, messages...)...),
		}

		logger.V(log.VWarn).Info(message)
		for _, msg := range messages {
			logger.V(log.VWarn).Info(msg)
		}
		return reconcile.Result{}, dltpod, nil // don't requeue
	}

	// Reconcile casc
	result, err = userConfiguration.ReconcileCasc()
	if err != nil {
		return reconcile.Result{}, dltpod, err
	}
	if result.Requeue {
		return result, dltpod, nil
	}

	// Reconcile seedjobs, backups
	result, err = userConfiguration.ReconcileOthers()
	if err != nil {
		return reconcile.Result{}, dltpod, err
	}
	if result.Requeue {
		return result, dltpod, nil
	}

	if dltpod.Status.UserConfigurationCompletedTime == nil {
		now := metav1.Now()
		dltpod.Status.UserConfigurationCompletedTime = &now
		err = r.Client.Status().Update(context.TODO(), dltpod)
		if err != nil {
			return reconcile.Result{}, dltpod, errors.WithStack(err)
		}
		message := fmt.Sprintf("User configuration phase is complete, took %s",
			dltpod.Status.UserConfigurationCompletedTime.Sub(dltpod.Status.ProvisionStartTime.Time))
		*r.NotificationEvents <- event.Event{
			DLTPod: *dltpod,
			Phase:  event.PhaseUser,
			Level:  v1alpha1.NotificationLevelInfo,
			Reason: reason.NewUserConfigurationComplete(reason.OperatorSource, []string{message}),
		}
		logger.Info(message)
	}
	return reconcile.Result{}, dltpod, nil
}

// TODO
func (r *DLTPodReconciler) setDefaults(dltpod *v1alpha1.DLTPod) (requeue bool, err error) {
	changed := false
	logger := logx.WithValues("cr", dltpod.Name)

	var dltpodContainer v1alpha1.Container
	if len(dltpod.Spec.Jobs) == 0 {
		changed = true
		dltpodContainer = v1alpha1.Container{Name: resources.dltpodMasterContainerName}
	} else {
		if dltpod.Spec.Jobs[0].Name != resources.dltpodMasterContainerName {
			return false, errors.Errorf("first container in spec.master.containers must be dltpod container with name '%s', please correct CR", resources.dltpodMasterContainerName)
		}
		dltpodContainer = dltpod.Spec.Jobs[0]
	}

	if len(dltpodContainer.Image) == 0 {
		logger.Info("Setting default dltpod master image: " + constants.DefaultdltpodMasterImage)
		changed = true
		dltpodContainer.Image = constants.DefaultdltpodMasterImage
		dltpodContainer.ImagePullPolicy = corev1.PullAlways
	}
	if len(dltpodContainer.ImagePullPolicy) == 0 {
		logger.Info(fmt.Sprintf("Setting default dltpod master image pull policy: %s", corev1.PullAlways))
		changed = true
		dltpodContainer.ImagePullPolicy = corev1.PullAlways
	}

	if len(dltpodContainer.Command) == 0 {
		logger.Info("Setting default dltpod container command")
		changed = true
		dltpodContainer.Command = resources.GetdltpodMasterContainerBaseCommand()
	}

	if isResourceRequirementsNotSet(dltpodContainer.Resources) {
		logger.Info("Setting default dltpod master container resource requirements")
		changed = true
		dltpodContainer.Resources = resources.NewResourceRequirements("1", "500Mi", "1500m", "3Gi")
	}
	if reflect.DeepEqual(dltpod.Spec.Service, v1alpha1.Service{}) {
		logger.Info("Setting default dltpod master service")
		changed = true
		var serviceType = corev1.ServiceTypeClusterIP
		if r.dltpodAPIConnectionSettings.UseNodePort {
			serviceType = corev1.ServiceTypeNodePort
		}
		dltpod.Spec.Service = v1alpha1.Service{
			Type: serviceType,
			Port: constants.DefaultHTTPPortInt32,
		}
	}
	if reflect.DeepEqual(dltpod.Spec.SlaveService, v1alpha1.Service{}) {
		logger.Info("Setting default dltpod slave service")
		changed = true
		dltpod.Spec.SlaveService = v1alpha1.Service{
			Type: corev1.ServiceTypeClusterIP,
			Port: constants.DefaultSlavePortInt32,
		}
	}
	if len(dltpod.Spec.Jobs) > 1 {
		for i, container := range dltpod.Spec.Jobs[1:] {
			if r.setDefaultsForContainer(dltpod, container.Name, i+1) {
				changed = true
			}
		}
	}
	if len(dltpod.Spec.Backup.ContainerName) > 0 && dltpod.Spec.Backup.Interval == 0 {
		logger.Info("Setting default backup interval")
		changed = true
		dltpod.Spec.Backup.Interval = 30
	}

	if len(dltpod.Spec.Jobs) == 0 || len(dltpod.Spec.Jobs) == 1 {
		dltpod.Spec.Jobs = []v1alpha1.Container{dltpodContainer}
	} else {
		nodltpodContainers := dltpod.Spec.Jobs[1:]
		containers := []v1alpha1.Container{dltpodContainer}
		containers = append(containers, nodltpodContainers...)
		dltpod.Spec.Jobs = containers
	}

	if reflect.DeepEqual(dltpod.Spec.dltpodAPISettings, v1alpha1.dltpodAPISettings{}) {
		logger.Info("Setting default dltpod API settings")
		changed = true
		dltpod.Spec.dltpodAPISettings = v1alpha1.dltpodAPISettings{AuthorizationStrategy: v1alpha1.CreateUserAuthorizationStrategy}
	}

	if dltpod.Spec.dltpodAPISettings.AuthorizationStrategy == "" {
		logger.Info("Setting default dltpod API settings authorization strategy")
		changed = true
		dltpod.Spec.dltpodAPISettings.AuthorizationStrategy = v1alpha1.CreateUserAuthorizationStrategy
	}

	if changed {
		return changed, errors.WithStack(r.Client.Update(context.TODO(), dltpod))
	}
	return changed, nil
}

// TODO
func (r *DLTPodReconciler) setDefaultsForContainer(dltpod *v1alpha1.dltpod, containerName string, containerIndex int) bool {
	changed := false
	logger := logx.WithValues("cr", dltpod.Name, "container", containerName)

	if len(dltpod.Spec.Jobs[containerIndex].ImagePullPolicy) == 0 {
		logger.Info(fmt.Sprintf("Setting default container image pull policy: %s", corev1.PullAlways))
		changed = true
		dltpod.Spec.Jobs[containerIndex].ImagePullPolicy = corev1.PullAlways
	}
	if isResourceRequirementsNotSet(dltpod.Spec.Jobs[containerIndex].Resources) {
		logger.Info("Setting default container resource requirements")
		changed = true
		dltpod.Spec.Jobs[containerIndex].Resources = resources.NewResourceRequirements("50m", "50Mi", "100m", "100Mi")
	}
	return changed
}

func isResourceRequirementsNotSet(requirements corev1.ResourceRequirements) bool {
	return reflect.DeepEqual(requirements, corev1.ResourceRequirements{})
}
