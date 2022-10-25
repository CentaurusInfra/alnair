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

package v1alpha1

import (
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// EDIT THIS FILE!  THIS IS SCAFFOLDING FOR YOU TO OWN!
// NOTE: json tags are required.  Any new fields you add must have json tags for the fields to be serialized.

// NotificationLevel defines the level of a Notification.
type NotificationLevel string

const (
	// NotificationLevelWarning - Only Warnings
	NotificationLevelWarning NotificationLevel = "warning"

	// NotificationLevelInfo - Only info
	NotificationLevelInfo NotificationLevel = "info"
)

// SecretRef is reference to Kubernetes secret.
type SecretRef struct {
	Name string `json:"name"`
}

// ConfigMapRef is reference to Kubernetes ConfigMap.
type ConfigMapRef struct {
	Name string `json:"name"`
}

type QoSConfigurations struct {
	// Whether client should use Alnair or s3
	// +optional
	UseCache bool `json:"usecache"`

	// Load data in a lazy way
	// +optional
	LazyLoading bool `json:"lazyloading"`
}

type DataSourceStruct struct {
	Name   string   `json:"name"`
	Bucket string   `json:"bucket"`
	Keys   []string `json:"keys,omitempty"`
}

// Container defines Kubernetes container attributes.
type Job struct {
	// Name of the container specified as a DNS_LABEL.
	// Each container in a pod must have a unique name (DNS_LABEL).
	Name string `json:"name"`

	// Docker image name.
	// More info: https://kubernetes.io/docs/concepts/containers/images
	Image string `json:"image"`

	// Connection and dataset information.
	DataSource DataSourceStruct `json:"datasource"`

	// QoS configuration of the job from ConfigMap
	// +optional
	ConfigurationsFromConfigMap ConfigMapRef `json:"configurationsfromconfigmap,omitempty"`

	// QoS configuration of the job from ConfigMap
	// +optional
	Configurations QoSConfigurations `json:"configurations,omitempty"`

	// Entrypoint array. Not executed within a shell.
	// The docker image's ENTRYPOINT is used if this is not provided.
	// Variable references $(VAR_NAME) are expanded using the container's environment. If a variable
	// cannot be resolved, the reference in the input string will be unchanged. The $(VAR_NAME) syntax
	// can be escaped with a double $$, ie: $$(VAR_NAME). Escaped references will never be expanded,
	// regardless of whether the variable exists or not.
	// More info: https://kubernetes.io/docs/tasks/inject-data-application/define-command-argument-container/#running-a-command-in-a-shell
	Command []string `json:"command"`

	// Image pull policy.
	// One of Always, Never, IfNotPresent.
	// Defaults to Always.
	// +optional
	ImagePullPolicy corev1.PullPolicy `json:"imagePullPolicy"`

	// Compute Resources required by this container.
	// More info: https://kubernetes.io/docs/concepts/configuration/manage-compute-resources-container/
	// +optional
	Resources corev1.ResourceRequirements `json:"resources"`

	// Arguments to the entrypoint.
	// The docker image's CMD is used if this is not provided.
	// Variable references $(VAR_NAME) are expanded using the container's environment. If a variable
	// cannot be resolved, the reference in the input string will be unchanged. The $(VAR_NAME) syntax
	// can be escaped with a double $$, ie: $$(VAR_NAME). Escaped references will never be expanded,
	// regardless of whether the variable exists or not.
	// More info: https://kubernetes.io/docs/tasks/inject-data-application/define-command-argument-container/#running-a-command-in-a-shell
	// +optional
	Args []string `json:"args,omitempty"`

	// Container's working directory.
	// If not specified, the container runtime's default will be used, which
	// might be configured in the container image.
	// +optional
	WorkingDir string `json:"workingDir,omitempty"`

	// List of ports to expose from the container. Exposing a port here gives
	// the system additional information about the network connections a
	// container uses, but is primarily informational. Not specifying a port here
	// DOES NOT prevent that port from being exposed. Any port which is
	// listening on the default "0.0.0.0" address inside a container will be
	// accessible from the network.
	// +optional
	Ports []corev1.ContainerPort `json:"ports,omitempty"`

	// List of sources to populate environment variables in the container.
	// The keys defined within a source must be a C_IDENTIFIER. All invalid keys
	// will be reported as an event when the container is starting. When a key exists in multiple
	// sources, the value associated with the last source will take precedence.
	// Values defined by an Env with a duplicate key will take precedence.
	// +optional
	EnvFrom []corev1.EnvFromSource `json:"envFrom,omitempty"`

	// List of environment variables to set in the container.
	// +optional
	Env []corev1.EnvVar `json:"env,omitempty"`

	// Pod volumes to mount into the container's filesystem.
	// +optional
	VolumeMounts []corev1.VolumeMount `json:"volumeMounts,omitempty"`

	// Actions that the management system should take in response to container lifecycle events.
	// +optional
	Lifecycle *corev1.Lifecycle `json:"lifecycle,omitempty"`

	// +optional
	TTY bool `json:"tty,omitempty"`

	// +optional
	Stdin bool `json:"stdin,omitempty"`
}

// AlnairPodSpec defines the desired state of AlnairPod
type AlnairPodSpec struct {
	// INSERT ADDITIONAL SPEC FIELDS - desired state of cluster
	// Important: Run "make" to regenerate code after modifying this file

	Secret SecretRef `json:"secret"`
	Jobs   []Job     `json:"jobs"`

	// List of volumes that can be mounted by containers belonging to the pod.
	// More info: https://kubernetes.io/docs/concepts/storage/volumes
	// +optional
	Volumes []corev1.Volume `json:"volumes,omitempty"`

	// Map of string keys and values that can be used to organize and categorize
	// (scope and select) objects. May match selectors of replication controllers
	// and services.
	// More info: http://kubernetes.io/docs/user-guide/labels
	// +optional
	Labels map[string]string `json:"labels,omitempty"`

	// NodeSelector is a selector which must be true for the pod to fit on a node.
	// Selector which must match a node's labels for the pod to be scheduled on that node.
	// More info: https://kubernetes.io/docs/concepts/configuration/assign-pod-node/
	// +optional
	NodeSelector map[string]string `json:"nodeSelector,omitempty"`

	NodeName string `json:"nodeName,omitempty"`

	// Host networking requested for this pod. Use the host's network namespace.
	// If this option is set, the ports that will be used must be specified. Default to false.
	// +optional
	HostNetwork bool `json:"hostnetwork,omitempty"`

	// Node sequence of downloading dataset
	// +optional
	NodePriority []string `json:"nodePriority,omitempty"`
}

// AlnairPodStatus defines the observed state of AlnairPod
type AlnairPodStatus struct {
	// Client container in this Pod
	// +optional
	Client string `json:"client,omitempty"`

	// Created job containers in this Pod
	// +optional
	CreatedJobs []string `json:"createdjobs,omitempty"`
}

//+kubebuilder:object:root=true
//+kubebuilder:subresource:status

// AlnairPod is the Schema for the alnairpods API
type AlnairPod struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   AlnairPodSpec   `json:"spec,omitempty"`
	Status AlnairPodStatus `json:"status,omitempty"`
}

//+kubebuilder:object:root=true

// AlnairPodList contains a list of AlnairPod
type AlnairPodList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []AlnairPod `json:"items"`
}

func init() {
	SchemeBuilder.Register(&AlnairPod{}, &AlnairPodList{})
}
