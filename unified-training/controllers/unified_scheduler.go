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
	"bytes"
	"context"
	aiv1alpha1 "elastictraining/api/v1alpha1"
	"fmt"
	"sort"
	"time"

	"github.com/go-logr/logr"
	"github.com/prometheus/common/log"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

const TIME_RESCHEDULING = 10 //time in minutes for a rescheduling
const TIME_REQUEUE = 10      //time in seconds for a failed requeue

// UnifiedSchedulerReconciler sets the TargetReplicas of a Unified Job
type UnifiedSchedulerReconciler struct {
	client.Client
	Log          logr.Logger
	Scheme       *runtime.Scheme
	timestamp    time.Time
	JobMultiNode map[aiv1alpha1.UnifiedJobType]bool //a map that checks if a job is able to be scheduled on multiple nodes
	primaryJob   string                             //primary used to prevent rescheduling cycles from occuring multiple times when there are multiple jobs
}

func (r *UnifiedSchedulerReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := r.Log.WithValues("UnifiedJob", req.NamespacedName)

	var ujob aiv1alpha1.UnifiedJob
	if err := r.Get(ctx, req.NamespacedName, &ujob); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	// to prevent rescheduling cycles from occuring multiple times when there are multiple jobs, assign a "primary" job
	isPrimary, err := r.CheckPrimaryJob(ctx, ujob)
	if err != nil {
		log.Info(fmt.Sprintf("Could not check primary job: %s", err.Error()))
		return ctrl.Result{Requeue: true}, nil
	}

	// rescheduling to find global optima
	// only reschedule if this is the primary job
	if isPrimary && time.Since(r.timestamp).Minutes() > TIME_RESCHEDULING {
		r.RescheduleJobs(ctx, ujob.Namespace)
		r.timestamp = time.Now()

		return ctrl.Result{RequeueAfter: TIME_REQUEUE * time.Second}, nil
	}

	if ujob.Spec.ReplicaSpec.TargetReplicas != nil {

		log.Info("Target replicas already specified, do nothing")
		// requeue for rescheduling
		// requeue even if not primary job as the primary job may change
		return ctrl.Result{RequeueAfter: TIME_REQUEUE * time.Second}, nil
	}

	if ujob.Status.UnifiedJobStatus != aiv1alpha1.JobWaiting {
		return ctrl.Result{}, nil
	}

	//only assign to new job if the job is waiting

	//criteria is 1. fulfills maxreplicas 2. num left
	currNodeConfig := r.AssignToNewJob(ctx, ujob)

	if len(currNodeConfig) == 0 {
		log.Info("No suitable node configuration found.")
		return ctrl.Result{RequeueAfter: TIME_REQUEUE * time.Second}, nil
	}

	if err := r.updateTargetReplicas(ctx, ujob, currNodeConfig); err != nil {
		log.Info(fmt.Sprintf("Error in updating Target Replicas: %s", err.Error()))
		return ctrl.Result{RequeueAfter: TIME_REQUEUE * time.Second}, nil
	}

	log.Info(fmt.Sprintf("Target Replicas for job %s updated to %s (nodeConfig)", ujob.Name, mapAsString(currNodeConfig)))

	// cannot find enough GPU even after preemption, wait for some time and try again
	return ctrl.Result{RequeueAfter: TIME_REQUEUE * time.Second}, nil
}

func (r *UnifiedSchedulerReconciler) SetupWithManager(mgr ctrl.Manager) error {
	//set up multinode map
	r.JobMultiNode = map[aiv1alpha1.UnifiedJobType]bool{
		aiv1alpha1.BasicJobType:          false,
		aiv1alpha1.ElasticPytorchJobType: true,
		aiv1alpha1.ElasticHorovodJobType: true,
	}

	r.primaryJob = ""

	//set up clock
	r.timestamp = time.Now()

	return ctrl.NewControllerManagedBy(mgr).
		For(&aiv1alpha1.UnifiedJob{}).
		Complete(r)
}

func (r *UnifiedSchedulerReconciler) CheckPrimaryJob(ctx context.Context, ujob aiv1alpha1.UnifiedJob) (bool, error) {
	// check if the current job is the primary job
	// if not, check if there is another primary job
	// if no other primary jobs, set as primary job

	if r.primaryJob == "" { // if no primary job, set curr one as
		r.primaryJob = ujob.Name
		log.Info(fmt.Sprintf("Changed primary job to %s", r.primaryJob))
	}

	if ujob.Name == r.primaryJob { // is primary job
		// switch primary job if the job will not be runing anymore
		if ujob.Status.UnifiedJobStatus == aiv1alpha1.JobCompleted || ujob.Status.UnifiedJobStatus == aiv1alpha1.JobFailed {
			changed, err := r.changePrimaryJob(ctx, ujob.Namespace)
			if err != nil {
				log.Info("Could not change primary job.")
				return false, err
			}

			if !changed {
				log.Info("Job lists are empty")
				return false, nil
			}

			log.Info(fmt.Sprintf("Changed primary job to %s", r.primaryJob))

			return false, nil
		}
		return true, nil
	}

	return false, nil

}

func (r *UnifiedSchedulerReconciler) RescheduleJobs(ctx context.Context, namespace string) {
	log.Info("Rescheduling Jobs")
	queuedJobs, runningJobs := r.BuildJobLists(ctx, []client.ListOption{client.InNamespace(namespace)})
	if len(queuedJobs) == 0 {
		return
	}
	reschedulableJobs := filterByReschedulable(runningJobs)
	allJobs := append(queuedJobs, reschedulableJobs...) //jobs to reschedule
	nameJobMap, nodeJobMap := nameToJob(allJobs), nodeToJobs(reschedulableJobs)

	allocatableNodeMap, allocatableKeys := r.AllocatableNodeMap(ctx)                          //idle gpus in this rescheduling cycle
	reschedulableNodeMap := addMaps(allocatableNodeMap, currJobOccupation(reschedulableJobs)) //node map for this cycle = allocatable + occupied by reschedulable

	changedJobs := []string{} // a slice of the changed jobs for later updating all at once

	for _, node := range allocatableKeys { // for all nodes that have idle GPUs currently, sorted from most free -> least free
		// simply go one-by-one through queuedJobs
		// try to assign a job in queue; do this by downsizing the rest of the jobs in the
		// currently occupied gpu is resched[node]-alloca[node], multinode jobs here are considered not rescehdulable
		jobsInNode := nameListToJobList(nodeJobMap[node], nameJobMap)
		rJobs, gpuDiff := removeMultiNodeJobs(jobsInNode, node)

		for _, job := range queuedJobs { // iterate through jobs to see if any fit in the node with idle gpu
			low, _ := gpuRange(append(rJobs, job))

			if *job.Spec.ReplicaSpec.MinReplicas < (allocatableNodeMap[node]) { //able to be directly scheduled, directly run assignnew

				nodeConfig := r.AssignToNewJob(ctx, job)
				for k, v := range nodeConfig {
					allocatableNodeMap[k] -= v
					nodeJobMap[k] = append(nodeJobMap[k], job.Name)
				}
				job.Spec.ReplicaSpec.TargetReplicas = nodeConfig
				changedJobs = append(changedJobs, job.Name)

				if allocatableNodeMap[node] == 0 { //if this was the node that was assigned to, break; otherwise, iterate through queuedjobs more
					break
				}

			} else if low < reschedulableNodeMap[node]-gpuDiff { // rebalanceNode will ensure no idle gpus
				changedJobs = append(changedJobs, rebalanceNode(allocatableNodeMap[node], rJobs, job, node, nameJobMap)...)
				break
			}
			//not enough resources, just go to next job
		}

	}

	for _, jobName := range changedJobs {
		r.updateTargetReplicas(ctx, nameJobMap[jobName], nameJobMap[jobName].Spec.ReplicaSpec.TargetReplicas)
	}
}

func (r *UnifiedSchedulerReconciler) TotalNodeMap(ctx context.Context) map[string]int64 {
	//returns a NodeName:numGpu map where numGpu is the total number of GPUs on the node (whether allocated or not)
	var nodeList corev1.NodeList
	_ = r.List(ctx, &nodeList)

	nodeMap := map[string]int64{}

	for _, node := range nodeList.Items {
		gpus := node.Status.Capacity["nvidia.com/gpu"]
		numGpus, ok := gpus.AsInt64()
		if ok {
			nodeMap[node.ObjectMeta.Name] = numGpus
		}
	}

	return nodeMap
}

func (r *UnifiedSchedulerReconciler) AllocatableNodeMap(ctx context.Context) (map[string]int64, []string) {
	//returns a NodeName:numGpu map where numGpu is the number of unoccupied GPUs on the node
	var nodeList corev1.NodeList
	_ = r.List(ctx, &nodeList)

	nodeMap := map[string]int64{}

	for _, node := range nodeList.Items {
		gpus := node.Status.Allocatable["nvidia.com/gpu"]
		numGpus, ok := gpus.AsInt64()
		if ok {
			nodeMap[node.ObjectMeta.Name] = numGpus
		}
	}

	//get sorted keys by some criteria (ie number of GPU)
	sortedKeyList := sortMapByValue(nodeMap)

	return nodeMap, sortedKeyList
}

func (r *UnifiedSchedulerReconciler) BuildJobLists(ctx context.Context, listOpts []client.ListOption) ([]aiv1alpha1.UnifiedJob, []aiv1alpha1.UnifiedJob) {
	//returns a list of 1. Queued jobs 2. Running jobs
	//might want one of migrating jobs?
	var queuedJobs []aiv1alpha1.UnifiedJob
	var runningJobs []aiv1alpha1.UnifiedJob

	var ujobList aiv1alpha1.UnifiedJobList
	_ = r.List(ctx, &ujobList, listOpts...)

	for _, ujob := range ujobList.Items {
		if ujob.Status.UnifiedJobStatus == aiv1alpha1.JobCompleted || ujob.Status.UnifiedJobStatus == aiv1alpha1.JobFailed {
			continue
		}

		if ujob.Status.UnifiedJobStatus == aiv1alpha1.JobWaiting {
			queuedJobs = append(queuedJobs, ujob)
		} else {
			runningJobs = append(runningJobs, ujob)
		}
	}

	return queuedJobs, runningJobs
}

func (r *UnifiedSchedulerReconciler) AssignToNewJob(ctx context.Context, ujob aiv1alpha1.UnifiedJob) map[string]int64 {
	//New job, find available resources
	log.Info("Looking for resources")

	//assign by a simple greedy algorithm
	// keep track of a gpusLeft; iterate through the nodes
	// if the node has gpus > gpusleft, allocate the rest of those gpus and leave function
	// otherwise, have gpusleft = gpusleft - currNodeGpuCount and iterate again
	nodeMap, sortedKeys := r.AllocatableNodeMap(ctx)

	if getTotalGPU(nodeMap) < int(*ujob.Spec.ReplicaSpec.MinReplicas) {
		return nil
	}

	nodeConfig := assignSimpleGreedy(nodeMap, sortedKeys, ujob, r.JobMultiNode[ujob.JobType])

	return nodeConfig

}

func assignSimpleGreedy(nodeMap map[string]int64, sortedKeys []string, ujob aiv1alpha1.UnifiedJob, multiNode bool) map[string]int64 {
	//assigns a nodeConfig (map) to the job that comes in
	currNodeConfig := make(map[string]int64)
	gpusLeft := *ujob.Spec.ReplicaSpec.MaxReplicas
	found := false

	// algorithm: (for looping from most gpus -> least gpus)
	// set gpusleft to maxreplicas
	// if there is a node with allocatable gpus > gpusleft, use the node with min allocatable gpus that is still > gpusleft (may change this for gpu:job affinity)
	// otherwise, set the config to the node with the greatest gpus (which is < gpusleft) and update gpusleft (gpusleft > 0)
	// -> if the config is acceptable (> minreplicas), finish (this is due to multinode affecting training greatly)
	// -> repeat algorithm if multiNode, exit if not
	//      -> if the ending is still not acceptable return nothing
	for _, nodeName := range sortedKeys {

		if nodeMap[nodeName] >= gpusLeft { //if node can fit rest of job
			currNodeConfig = map[string]int64{nodeName: gpusLeft} //reset currNodeConfig
			found = true
		} else { //more partial allocation required
			if found { //already fulfilled max
				break
			}
			if !multiNode {
				if nodeMap[nodeName] > *ujob.Spec.ReplicaSpec.MinReplicas {
					currNodeConfig[nodeName] = nodeMap[nodeName]
				}
				break
			}
			//rest of the numGpus are < maxreplicas
			currNodeConfig[nodeName] = nodeMap[nodeName]
			gpusLeft -= nodeMap[nodeName]
			//if current is already greater than min (avoid multi-noding)
			if getTotalGPU(currNodeConfig) > int(*ujob.Spec.ReplicaSpec.MinReplicas) {
				break
			}
		}
	}

	if getTotalGPU(currNodeConfig) < int(*ujob.Spec.ReplicaSpec.MinReplicas) { //for loop exited, check if nodeconfig is acceptable
		return make(map[string]int64)
	} else {
		return currNodeConfig
	}
}

func (r *UnifiedSchedulerReconciler) updateTargetReplicas(ctx context.Context, ujob aiv1alpha1.UnifiedJob, nodeConfig map[string]int64) error {
	// update target replicas to specified nodeconfig
	ujob.Spec.ReplicaSpec.TargetReplicas = nodeConfig
	return r.Update(ctx, &ujob)
}

func (r *UnifiedSchedulerReconciler) changePrimaryJob(ctx context.Context, namespace string) (bool, error) {
	// choose random job to change to (preferably lasts as long as possible)
	queuedJobs, runningJobs := r.BuildJobLists(ctx, []client.ListOption{client.InNamespace(namespace)})
	changed := true

	if len(queuedJobs) > 0 {
		r.primaryJob = queuedJobs[0].Name
	} else if len(runningJobs) > 0 {
		r.primaryJob = runningJobs[0].Name
	} else {
		r.primaryJob = ""
		changed = false
	}

	return changed, nil

}

// helper functions ----------------------------------------------------

//new struct definition for custom sorting

type kvPair struct {
	key   string
	value int64
}

type keySorter []kvPair

func (k keySorter) Len() int           { return len(k) }
func (k keySorter) Swap(i, j int)      { k[i], k[j] = k[j], k[i] }
func (k keySorter) Less(i, j int) bool { return k[i].value > k[j].value }

func sortMapByValue(nodeMap map[string]int64) []string {
	var sortedKeys keySorter = make([]kvPair, len(nodeMap))
	sortedKeyList := make([]string, len(nodeMap))
	i := 0
	for k := range nodeMap {
		sortedKeys[i] = kvPair{k, nodeMap[k]}
		i++
	}

	sort.Sort(sortedKeys)

	for i, kv := range sortedKeys {
		sortedKeyList[i] = kv.key
	}

	return sortedKeyList
}

func mapAsString(m map[string]int64) string {
	b := new(bytes.Buffer)
	fmt.Fprint(b, "{")
	for k, v := range m {
		fmt.Fprintf(b, "%s : %d, ", k, v)
	}
	fmt.Fprint(b, "}")
	return b.String()
}

func gpuRange(ujobList []aiv1alpha1.UnifiedJob) (int64, int64) {
	//return the min and max number of gpus a list of jobs can allocate
	minGpu := int64(0)
	maxGpu := int64(0)
	for _, ujob := range ujobList {
		minGpu += *ujob.Spec.ReplicaSpec.MinReplicas
		maxGpu += *ujob.Spec.ReplicaSpec.MaxReplicas
	}

	return minGpu, maxGpu
}

func filterByReschedulable(jobList []aiv1alpha1.UnifiedJob) []aiv1alpha1.UnifiedJob {
	//filter a jobList by a job being reschedulable or not
	var reschedulableJobList []aiv1alpha1.UnifiedJob

	for _, ujob := range jobList {
		if ujob.Spec.Reschedulable {
			reschedulableJobList = append(reschedulableJobList, ujob)
		}
	}

	return reschedulableJobList
}

func currJobOccupation(jobList []aiv1alpha1.UnifiedJob) map[string]int64 {
	//get the gpu occupation of a list of jobs
	nodeMap := make(map[string]int64)

	for _, ujob := range jobList {
		for nodeName, numGpu := range ujob.Spec.ReplicaSpec.TargetReplicas {
			nodeMap[nodeName] += numGpu
		}
	}

	return nodeMap
}

func addMaps(map1 map[string]int64, map2 map[string]int64) map[string]int64 {
	//adds two maps together
	map3 := make(map[string]int64)

	for k, v := range map1 {
		map3[k] += v
	}

	for k, v := range map2 {
		map3[k] += v
	}

	return map3

}

func nameToJob(jobList []aiv1alpha1.UnifiedJob) map[string]aiv1alpha1.UnifiedJob {
	// build map of jobNmae : job
	nameJobMap := make(map[string]aiv1alpha1.UnifiedJob)
	for _, ujob := range jobList {
		nameJobMap[ujob.Name] = ujob
	}

	return nameJobMap
}

func nodeToJobs(jobList []aiv1alpha1.UnifiedJob) map[string][]string {
	// build map of node : []jobName
	nodeJobMap := make(map[string][]string)

	for _, ujob := range jobList {
		for nodeName := range ujob.Spec.ReplicaSpec.TargetReplicas {
			nodeJobMap[nodeName] = append(nodeJobMap[nodeName], ujob.Name)
		}
	}

	return nodeJobMap
}

func nameListToJobList(nameList []string, nameJobMap map[string]aiv1alpha1.UnifiedJob) []aiv1alpha1.UnifiedJob {
	// convert []jobNmae to []job
	jobList := []aiv1alpha1.UnifiedJob{}

	for _, name := range nameList {
		jobList = append(jobList, nameJobMap[name])
	}

	return jobList

}

func removeMultiNodeJobs(jobList []aiv1alpha1.UnifiedJob, nodeName string) ([]aiv1alpha1.UnifiedJob, int64) {
	// remove multi node jobs from list for simplicity, returns the gpu diff
	filteredJobList := []aiv1alpha1.UnifiedJob{}
	gpuDiff := int64(0)

	for _, job := range jobList {
		if len(job.Spec.ReplicaSpec.TargetReplicas) == 1 {
			filteredJobList = append(filteredJobList, job)
		} else {
			gpuDiff += job.Spec.ReplicaSpec.TargetReplicas[nodeName]
		}
	}

	return filteredJobList, gpuDiff
}

func rebalanceNode(numGpu int64, currJobs []aiv1alpha1.UnifiedJob, incomingJob aiv1alpha1.UnifiedJob, nodeName string, nameJobMap map[string]aiv1alpha1.UnifiedJob) []string {
	// numGpu is the current number of free GPU (need to deallocate at least newjob.minreplicas - numGpu)
	// return the jobs that were changed
	// rebalances node to include job (downsize currJobs)

	// check how much downsizing is available at this time (use targetreplicas)
	diffMap := map[string]int64{}

	for _, job := range currJobs { // build diff map; if any of the replicaspec is currently greater than min, calculate the diff
		if *job.Spec.ReplicaSpec.MinReplicas < job.Spec.ReplicaSpec.TargetReplicas[nodeName] {
			diffMap[job.Name] = job.Spec.ReplicaSpec.TargetReplicas[nodeName] - *job.Spec.ReplicaSpec.MinReplicas
		}
	}

	// sort the map by diffs (minimize the amount of jobs that are downsized)
	sortedDiffJobs := sortMapByValue(diffMap)

	changedJobs := []string{incomingJob.Name}

	gpuToFree := *incomingJob.Spec.ReplicaSpec.MinReplicas - numGpu
	for _, jobName := range sortedDiffJobs {
		changedJobs = append(changedJobs, jobName)
		if diffMap[jobName] >= gpuToFree {
			nameJobMap[jobName].Spec.ReplicaSpec.TargetReplicas[nodeName] -= gpuToFree
			incomingJob.Spec.ReplicaSpec.TargetReplicas[nodeName] = *incomingJob.Spec.ReplicaSpec.MinReplicas
			break
		} else {
			nameJobMap[jobName].Spec.ReplicaSpec.TargetReplicas[nodeName] -= diffMap[jobName]
			gpuToFree -= diffMap[jobName]
		}
	}

	return changedJobs

}

func removeIndex(l []interface{}, i int) []interface{} {
	// remove element at index i, with bounds checking.
	// solution potentially slow but scale should be small vs. more undefined, fast solution
	if i == len(l)-1 {
		return l[:i]
	}

	return append(l[:i], l[i+1:]...)
}
