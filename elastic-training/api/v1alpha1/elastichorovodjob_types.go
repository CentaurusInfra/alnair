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

// ElasticHorovodJobSpec defines the desired state of ElasticHorovodJob
type ElasticHorovodJobSpec struct {
	LauncherSpec HorovodLauncherSpec       `json:"launcherSpec"`
	WorkersSpec  ElasticHorovodWorkersSpec `json:"workersSpec"`
}

type ElasticHorovodWorkersSpec struct {
	Image string `json:"image"`

	// +optional
	// +kubebuilder:validation:Minimum=0
	TargetReplicas *int32 `json:"targetReplicas,omitempty"`

	// +kubebuilder:validation:Minimum=0
	MinReplicas *int32 `json:"minReplicas,omitempty"`

	// +kubebuilder:validation:Minimum=0
	MaxReplicas *int32 `json:"maxReplicas,omitempty"`
}

// ElasticHorovodJobStatus defines the observed state of ElasticHorovodJob
type ElasticHorovodJobStatus struct {
	AvailableWorkerReplicas int32  `json:"availableReplicas"`
	Launcher                string `json:"launcher"`
	Workers                 string `json:"workers"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status

// ElasticHorovodJob is the Schema for the elastichorovodjobs API
type ElasticHorovodJob struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   ElasticHorovodJobSpec   `json:"spec,omitempty"`
	Status ElasticHorovodJobStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// ElasticHorovodJobList contains a list of ElasticHorovodJob
type ElasticHorovodJobList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []ElasticHorovodJob `json:"items"`
}

func init() {
	SchemeBuilder.Register(&ElasticHorovodJob{}, &ElasticHorovodJobList{})
}
