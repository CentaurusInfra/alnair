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

package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

type UnifiedJobType string

const (
	BasicJobType          UnifiedJobType = "BaseJob"
	ElasticHorovodJobType UnifiedJobType = "EHJob"
	ElasticPytorchJobType UnifiedJobType = "EPJob"
	HorovodJobType        UnifiedJobType = "HJob"
)

type UnifiedJobStatusType string

const (
	JobWaiting   UnifiedJobStatusType = "Waiting"
	JobRunning   UnifiedJobStatusType = "Running"
	JobPaused    UnifiedJobStatusType = "Paused"
	JobCompleted UnifiedJobStatusType = "Complete"
	JobFailed    UnifiedJobStatusType = "Failed"
	JobPending   UnifiedJobStatusType = "Pending"
	JobMigrating UnifiedJobStatusType = "Migrating"
)

// UnifiedJobSpec defines the desired state of UnifiedJobSpec
type UnifiedJobSpec struct {
	ReplicaSpec UnifiedJobReplicaSpec `json:"replicaSpec,omitempty"`
	JobSpec     UnifiedJobWorkersSpec `json:"jobSpec,omitempty"`

	Reschedulable bool `json:"reschedulable"`
}

type UnifiedJobReplicaSpec struct {
	// +optional
	TargetReplicas map[string]int64 `json:"targetReplicas,omitempty"`

	// +kubebuilder:validation:Minimum=0
	MinReplicas *int64 `json:"minReplicas,omitempty"`

	// +kubebuilder:validation:Minimum=0
	MaxReplicas *int64 `json:"maxReplicas,omitempty"`
}

type UnifiedJobWorkersSpec struct {
	Image       string   `json:"image"`
	UnifiedArgs []string `json:"unifiedArgs"`
}

// UnifiedJobStatus defines the observed state of UnifiedJob
type UnifiedJobStatus struct {
	UnifiedJobStatus UnifiedJobStatusType `json:"jobStatus"`
	ItersCompleted   int64                `json:"itersCompleted"`
	ItersTotal       int64                `json:"itersTotal"`
}

// +kubebuilder:object:root=true
// +kubebuilder:resource:shortName=ujob
// +kubebuilder:subresource:status

// UnifiedJob is the Schema for the UnifiedJob API
type UnifiedJob struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	JobType UnifiedJobType `json:"jobType,omitempty"`

	Spec   UnifiedJobSpec   `json:"spec,omitempty"`
	Status UnifiedJobStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// UnifiedJobList contains a list of UnifiedJob
type UnifiedJobList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []UnifiedJob `json:"items"`
}

func init() {
	SchemeBuilder.Register(&UnifiedJob{}, &UnifiedJobList{})
}
