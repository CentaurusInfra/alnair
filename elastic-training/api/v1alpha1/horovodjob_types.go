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

// HorovodJobSpec defines the desired state of HorovodJob
type HorovodJobSpec struct {
	LauncherSpec HorovodLauncherSpec `json:"launcherSpec"`
	WorkersSpec  HorovodWorkersSpec  `json:"workersSpec"`
}

type HorovodLauncherSpec struct {
	Image         string   `json:"image"`
	PythonCommand []string `json:"pythonCommand"`
}

type HorovodWorkersSpec struct {
	Image string `json:"image"`

	// +kubebuilder:validation:Minimum=0
	Replicas *int32 `json:"replicas,omitempty"`
}

// HorovodJobStatus defines the observed state of HorovodJob
type HorovodJobStatus struct {
	AvailableWorkerReplicas int32  `json:"availableReplicas"`
	Launcher                string `json:"launcher"`
	Workers                 string `json:"workers"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status

// HorovodJob is the Schema for the horovodjobs API
type HorovodJob struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   HorovodJobSpec   `json:"spec,omitempty"`
	Status HorovodJobStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// HorovodJobList contains a list of HorovodJob
type HorovodJobList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []HorovodJob `json:"items"`
}

func init() {
	SchemeBuilder.Register(&HorovodJob{}, &HorovodJobList{})
}
