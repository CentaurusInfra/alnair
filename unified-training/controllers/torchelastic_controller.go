package controllers

import (
	"context"
	aiv1alpha1 "elastictraining/api/v1alpha1"
	"fmt"
	"reflect"
	"strings"
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	corev1 "k8s.io/api/core/v1"
)

type TorchElasticJobController struct {
	etcdSvcName    string
	etcdServerName string
	jobName        string
	workerName     string
	workerSvcName  string
	launcherName   string
	jobID          string
}

func (r TorchElasticJobController) Test() string {
	return "TorchElasticJobController"
}

func (r TorchElasticJobController) UpdateStatus(reconciler *UnifiedJobReconciler, ctx context.Context, ujob aiv1alpha1.UnifiedJob, applyOpts []client.PatchOption) (bool, error) {
	oldStatus := ujob.Status.UnifiedJobStatus
	var newStatus aiv1alpha1.UnifiedJobStatusType

	if ujob.Spec.ReplicaSpec.TargetReplicas == nil {
		newStatus = aiv1alpha1.JobWaiting
	} else {
		workers, err := r.getWorkers(reconciler, ctx, ujob)
		currNodeMap := getCurrentNodeConfig(workers)
		if err != nil {
			if errors.IsNotFound(err) {
				//workers have not been created yet and targetreplicas was updated; this means job is pending and workers will be created in same reconcile cycle
				newStatus = aiv1alpha1.JobPending
				reconciler.Log.Info("Workers not created yet")
			} else {
				reconciler.Log.Info(fmt.Sprintf("Error in querying workers: %s", err.Error()))
			}
		} else if reflect.DeepEqual(currNodeMap, ujob.Spec.ReplicaSpec.TargetReplicas) {
			numPods := len(currNodeMap)

			//workers exist and match targetreplicas (desiredTotalGPU == len(workers)); these pods are either pending, running, failed, or completed

			//create map of status: num pods
			statusCount := map[corev1.PodPhase]int{
				corev1.PodPending:   0,
				corev1.PodRunning:   0,
				corev1.PodSucceeded: 0,
				corev1.PodFailed:    0,
				corev1.PodUnknown:   0,
			}

			for _, pod := range workers { //admission error = pending
				if podPending(pod) {
					statusCount[corev1.PodPending] += 1
				} else {
					statusCount[pod.Status.Phase] += 1
				}
			}

			// TODO: less naive status updating
			if statusCount[corev1.PodSucceeded] == numPods {
				newStatus = aiv1alpha1.JobCompleted
			} else if statusCount[corev1.PodPending] != 0 {
				newStatus = aiv1alpha1.JobPending
			} else if statusCount[corev1.PodFailed] != 0 {
				reconciler.Log.Info(fmt.Sprintf("Pod Error is %s", workers[0].Status.Reason))
				newStatus = aiv1alpha1.JobFailed
			} else if statusCount[corev1.PodRunning] == numPods {
				newStatus = aiv1alpha1.JobRunning
			}
		} else if oldStatus == aiv1alpha1.JobRunning || oldStatus == aiv1alpha1.JobMigrating {
			newStatus = aiv1alpha1.JobMigrating
		} else if oldStatus == aiv1alpha1.JobWaiting || oldStatus == aiv1alpha1.JobPending || oldStatus == "" {
			//job pending if some workers are missing
			newStatus = aiv1alpha1.JobPending
		} else {

			reconciler.Log.Info("Undefined; somehow mismatched")
			newStatus = aiv1alpha1.JobFailed

		}

	}

	changed := (newStatus != oldStatus)
	ujob.Status.UnifiedJobStatus = newStatus

	return changed, reconciler.Status().Update(ctx, &ujob)

}

func (r TorchElasticJobController) ReleaseResources(reconciler *UnifiedJobReconciler, ctx context.Context, ujob aiv1alpha1.UnifiedJob, deleteOpts []client.DeleteOption) error {
	//only workers to delete

	//deleteworkers includes the services
	if err := r.deleteWorkers(reconciler, ctx, ujob, deleteOpts); err != nil {
		return err
	}

	return nil
}

func (r TorchElasticJobController) DeleteAll(reconciler *UnifiedJobReconciler, ctx context.Context, ujob aiv1alpha1.UnifiedJob, deleteOpts []client.DeleteOption) error {
	//same as releaseresources
	//etcd server should persist
	return r.ReleaseResources(reconciler, ctx, ujob, deleteOpts)
}

func (r TorchElasticJobController) ServiceExists(reconciler *UnifiedJobReconciler, ctx context.Context, ujob aiv1alpha1.UnifiedJob) bool {
	// these are the etcd server and services; different from the headless services that the pods require

	if _, err := reconciler.getService(ctx, r.etcdSvcName, ujob.Namespace); err != nil {
		return false
	}

	if _, err := reconciler.getPod(ctx, r.etcdServerName, ujob.Namespace); err != nil {
		return false
	}

	return true
}

func (r TorchElasticJobController) CreateService(reconciler *UnifiedJobReconciler, ctx context.Context, applyOpts []client.PatchOption, ujob aiv1alpha1.UnifiedJob) error {
	// creates etcd service and server; different from headless service
	svc, svcPod, err := r.desiredService(reconciler, ctx, ujob)
	if err != nil {
		reconciler.Log.Info(fmt.Sprintf("Error in declaring Service: %s", err.Error()))
		return err
	}

	if err := reconciler.Patch(ctx, &svc, client.Apply, applyOpts...); err != nil {
		reconciler.Log.Info(fmt.Sprintf("Error in patching etcd service: %s", err.Error()))
		return err
	}

	if err := reconciler.Patch(ctx, &svcPod, client.Apply, applyOpts...); err != nil {
		reconciler.Log.Info(fmt.Sprintf("Error in patching etcd server: %s", err.Error()))
		return err
	}

	return nil
}

func (r TorchElasticJobController) StuckInPending(reconciler *UnifiedJobReconciler, ctx context.Context, ujob aiv1alpha1.UnifiedJob) bool {
	//TODO: check if any pod is in pending for > THRESH_TIME

	THRESH_TIME := time.Duration(10 * time.Second)

	for i := 1; i <= 2; i++ {
		//need to getworkers inside the loop to update the worker information
		podList, err := r.getWorkers(reconciler, ctx, ujob)

		if err != nil {
			if errors.IsNotFound(err) {
				return false
			}
			reconciler.Log.Info(fmt.Sprintf("Could not find workers: %s", err.Error()))
			return false
		}

		for _, pod := range podList {
			if podPending(pod) {
				if i == 1 {
					time.Sleep(THRESH_TIME)
					break
				}
				return true
			}
		}

		if i == 2 {
			return false
		}

	}

	return false //unreachable

}

func (r TorchElasticJobController) PatchAll(reconciler *UnifiedJobReconciler, ctx context.Context, ujob aiv1alpha1.UnifiedJob, applyOpts []client.PatchOption) error {
	//create and patch the job
	m := ujob.Spec.ReplicaSpec.TargetReplicas

	podList, svcList, err := r.desiredWorkers(reconciler, ctx, ujob, m)
	if err != nil {
		reconciler.Log.Info(fmt.Sprintf("Error in creating workers: %s", err.Error()))
		return err
	}

	for index := range podList {
		if err := reconciler.Patch(ctx, &podList[index], client.Apply, applyOpts...); err != nil {
			reconciler.Log.Info(fmt.Sprintf("Error in patching workers: %s", err.Error()))
			return err
		}

		if err := reconciler.Patch(ctx, &svcList[index], client.Apply, applyOpts...); err != nil {
			reconciler.Log.Info(fmt.Sprintf("Error in patching services: %s", err.Error()))
			return err
		}
	}

	ready, err := r.waitUntilWorkersReady(reconciler, ctx, ujob)
	if err != nil {
		return err
	}

	if !ready {
		reconciler.Log.Info("Workers are unable to be allocated")
	} else {
		reconciler.Log.Info("Workers are all running")
	}

	return nil
}

//DESIRED OBJECTS ------------------------------------------------------

func (r TorchElasticJobController) desiredService(reconciler *UnifiedJobReconciler, ctx context.Context, ujob aiv1alpha1.UnifiedJob) (corev1.Service, corev1.Pod, error) {
	// creates service and pod for etcd server
	svcName := r.etcdSvcName
	etcdName := r.etcdServerName
	svc := corev1.Service{
		TypeMeta: metav1.TypeMeta{APIVersion: corev1.SchemeGroupVersion.String(), Kind: "Service"},
		ObjectMeta: metav1.ObjectMeta{
			Name:      svcName,
			Namespace: ujob.Namespace,
		},
		Spec: corev1.ServiceSpec{
			Ports: []corev1.ServicePort{
				{
					Name:     "etcd-client-port",
					Port:     2379,
					Protocol: "TCP",
				},
			},
			Selector: map[string]string{"app": "etcd"},
		},
	}

	svcPod := corev1.Pod{
		TypeMeta: metav1.TypeMeta{APIVersion: corev1.SchemeGroupVersion.String(), Kind: "Pod"},
		ObjectMeta: metav1.ObjectMeta{
			Name:      etcdName,
			Namespace: ujob.Namespace,
			Labels:    map[string]string{"app": etcdName},
		},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{
					Name:  etcdName,
					Image: "quay.io/coreos/etcd:latest",
					Command: []string{
						"usr/local/bin/etcd",
						"--data-dir",
						"/var/lib/etcd",
						"--enable-v2",
						"--listen-client-urls",
						"http://0.0.0.0:2379",
						"--advertise-client-urls",
						"http://0.0.0.0:2379",
						"--initial-cluster-state",
						"new",
					},
					Ports: []corev1.ContainerPort{
						{
							Name:          "client",
							ContainerPort: 2379,
							Protocol:      "TCP",
						},
						{
							Name:          "server",
							ContainerPort: 2380,
							Protocol:      "TCP",
						},
					},
				},
			},
			//RestartPolicy: corev1.RestartPolicyAlways,
		},
	}

	// labels := map[string]string{
	// 	"UnifiedEPTJob": ujob.Name,
	// 	"role":          "worker",
	// }

	// //do not setControllerReference as we do not want server to be deleted when epjob is
	// workerService := corev1.Service{
	// 	TypeMeta: metav1.TypeMeta{APIVersion: corev1.SchemeGroupVersion.String(), Kind: "Service"},
	// 	ObjectMeta: metav1.ObjectMeta{
	// 		Name:      fmt.Sprintf(r.workerSvcName, ujob.Name),
	// 		Namespace: ujob.Namespace,
	// 	},
	// 	Spec: corev1.ServiceSpec{
	// 		ClusterIP: "None",
	// 		Selector:  labels,
	// 	},
	// }

	// if err := ctrl.SetControllerReference(&ujob, &workerService, reconciler.Scheme); err != nil {
	// 	return svc, svcPod, nil
	// }

	return svc, svcPod, nil
}

func (r TorchElasticJobController) desiredWorkers(reconciler *UnifiedJobReconciler, ctx context.Context, ujob aiv1alpha1.UnifiedJob, nodeMap map[string]int64) ([]corev1.Pod, []corev1.Service, error) {
	//returns workers as a list of pods that each take 1 GPU
	//all the workers are running the command independently

	svcName := r.etcdSvcName
	//svc, err := r.getService(reconciler, ctx, svcName, ujob.Namespace)
	_, err := reconciler.getService(ctx, svcName, ujob.Namespace)
	if err != nil {
		reconciler.Log.Info("Service not available for TorchElastic.")
		return []corev1.Pod{}, []corev1.Service{}, err
	}

	podNum := getTotalGPU(nodeMap)
	podList := make([]corev1.Pod, podNum)
	svcList := make([]corev1.Service, podNum)
	index := 0

	//iterate over nodename, numGpu to create one pod per GPU
	for nodeName, numGpu := range nodeMap {
		podCommand := r.genPodCommand(ujob, len(nodeMap), numGpu)
		labels := r.genLabels(ujob, []string{"UnifiedEPTJob", "role"})
		labels["workerID"] = fmt.Sprintf("%d", index)

		currPod, currSvc, err := r.desiredPod(reconciler, ctx, ujob, podCommand, labels, index, numGpu, nodeName)
		if err != nil {
			reconciler.Log.Info(fmt.Sprintf("Pod on %s with %d GPU unable to be created: %s", nodeName, numGpu, err.Error()))
		}

		podList[index] = currPod
		svcList[index] = currSvc
		index += 1

	}

	return podList, svcList, err
}

func (r TorchElasticJobController) desiredPod(reconciler *UnifiedJobReconciler, ctx context.Context, ujob aiv1alpha1.UnifiedJob, podCommand string,
	labels map[string]string, nodeIndex int, numGpu int64, nodeName string) (corev1.Pod, corev1.Service, error) {

	//returns pod and headless service for pod

	pod := corev1.Pod{
		TypeMeta: metav1.TypeMeta{APIVersion: corev1.SchemeGroupVersion.String(), Kind: "Pod"},
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf(r.workerName, ujob.Name, nodeIndex),
			Namespace: ujob.Namespace,
			Labels:    labels,
		},
		Spec: corev1.PodSpec{
			SchedulerName: "default-scheduler", //change if adding other schedulers
			//may want shared memory volume + container if large dataloaders
			Containers: []corev1.Container{
				{
					Name:    "worker",
					Image:   ujob.Spec.JobSpec.Image,
					Command: []string{"/bin/sh"},
					Args: []string{
						"-c",
						podCommand,
					},
					Resources: corev1.ResourceRequirements{
						Limits: corev1.ResourceList{
							corev1.ResourceName("nvidia.com/gpu"): *resource.NewQuantity(numGpu, resource.DecimalSI),
						},
					},
					Ports: []corev1.ContainerPort{
						{
							ContainerPort: 22,
							Protocol:      "TCP",
						},
					},
				},
			},
			RestartPolicy: corev1.RestartPolicyNever,
			NodeName:      nodeName,
		},
	}

	//create headless service bound to node as well
	svc := corev1.Service{
		TypeMeta: metav1.TypeMeta{APIVersion: corev1.SchemeGroupVersion.String(), Kind: "Service"},
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf(r.workerName, ujob.Name, nodeIndex),
			Namespace: ujob.Namespace,
		},
		Spec: corev1.ServiceSpec{
			ClusterIP: "None",
			Selector:  labels,
		},
	}

	if err := ctrl.SetControllerReference(&ujob, &pod, reconciler.Scheme); err != nil {
		return pod, svc, err
	}

	if err := ctrl.SetControllerReference(&ujob, &svc, reconciler.Scheme); err != nil {
		return pod, svc, err
	}

	return pod, svc, nil

}

//GET OBJECTS	------------------------------------------------------

func (r TorchElasticJobController) getWorkers(reconciler *UnifiedJobReconciler, ctx context.Context, ujob aiv1alpha1.UnifiedJob) ([]corev1.Pod, error) {
	labels := r.genLabels(ujob, []string{"UnifiedEPTJob", "role"})

	return reconciler.getPodsByLabel(ctx, ujob.Namespace, labels)
}

//DELETE OBJECTS------------------------------------------------------

func (r TorchElasticJobController) deleteWorkers(reconciler *UnifiedJobReconciler, ctx context.Context, ujob aiv1alpha1.UnifiedJob, deleteOpts []client.DeleteOption) error {
	//deletes workers and headless services for the workers
	workers, err := r.getWorkers(reconciler, ctx, ujob)
	if err != nil {
		reconciler.Log.Info("Unable to obtain Workers (Pod List)")
		return nil
	}

	for _, pod := range workers {
		if err := reconciler.Delete(ctx, &pod, deleteOpts...); err != nil {
			reconciler.Log.Info(fmt.Sprintf("Unable to delete pod %s", pod.Name))
			return err
		}

		//retrieve headless service
		podSvc, err := reconciler.getService(ctx, fmt.Sprintf(r.workerSvcName, ujob.Name), ujob.Namespace)
		if err != nil {
			reconciler.Log.Info(fmt.Sprintf("Unable to find Pod Service for pod %s", pod.Name))
			return err
		}

		if err := reconciler.Delete(ctx, &podSvc, deleteOpts...); err != nil {
			reconciler.Log.Info(fmt.Sprintf("Unable to delete pod service %s", podSvc.Name))
			return err
		}
	}

	return nil
}

//HELPER FUNCTIONS---------------------------------------------------

func (r TorchElasticJobController) genLabels(ujob aiv1alpha1.UnifiedJob, keys []string) map[string]string {
	// generate labels based on keys
	allLabels := map[string]string{
		"UnifiedEPTJob": ujob.Name,
		"role":          "worker",
	}

	labels := map[string]string{}

	for _, key := range keys {
		labels[key] = allLabels[key]
	}

	return allLabels
}

func (r TorchElasticJobController) genPodCommand(ujob aiv1alpha1.UnifiedJob, numNodes int, numGpu int64) string {
	//generate individual pod command for each pod
	jobID := fmt.Sprintf(r.jobID, ujob.Name)
	port := 2379
	rdzv_id := jobID
	rdzv_backend := "etcd"
	//rdzv_endpoint := fmt.Sprintf("%s:%s", svc.Spec.ClusterIP, port)
	rdzv_endpoint := fmt.Sprintf("torchelastic-etcd-service:%d", port)

	launchCommand := "python -m torchelastic.distributed.launch"
	launchArgs := fmt.Sprintf("--nnodes 1:%d --nproc_per_node %d --rdzv_id %s --rdzv_backend %s --rdzv_endpoint %s",
		numNodes, numGpu, rdzv_id, rdzv_backend, rdzv_endpoint)
	pythonCommand := strings.Join(ujob.Spec.JobSpec.UnifiedArgs, " ")
	podCommand := fmt.Sprintf("%s %s %s", launchCommand, launchArgs, pythonCommand)

	return podCommand
}

func (r TorchElasticJobController) areWorkersRunning(podList []corev1.Pod, nodeConfig map[string]int64) bool {
	if reflect.DeepEqual(getCurrentNodeConfig(podList), nodeConfig) {
		return false
	}
	for _, pod := range podList {
		if pod.Status.Phase != corev1.PodRunning {
			return false
		}
	}
	return true
}

func (r TorchElasticJobController) waitUntilWorkersReady(reconciler *UnifiedJobReconciler, ctx context.Context, ujob aiv1alpha1.UnifiedJob) (bool, error) {
	time.Sleep(1 * time.Second)
	startTime := time.Now()
	nodeConfig := ujob.Spec.ReplicaSpec.TargetReplicas
	for {
		if time.Since(startTime) > 30*time.Second {
			break
		}
		podList, err := r.getWorkers(reconciler, ctx, ujob)
		if err != nil {
			if errors.IsNotFound(err) {
				time.Sleep(1 * time.Second)
				continue
			}
			reconciler.Log.Info(fmt.Sprintf("Error in fetching workers: %s", err.Error()))
			return false, err
		}
		if r.areWorkersRunning(podList, nodeConfig) {
			return true, nil
		}
		time.Sleep(1 * time.Second)
	}

	return false, nil
}
