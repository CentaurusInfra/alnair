# UnifiedJob

UnifiedJob is a Kubernetes [Custom Resource Defintion](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/) (CRD). It allows for the Scheduler and Controller to act on a single object (the ujob), but also simplifies the user experience to only their job's essentials (file and amount of GPU resources). The full API 

## UnifiedJob API 

The API is defined as follows: 


```yaml
apiVersion: ai.centauruscloud.io/v1alpha1
kind: UnifiedJob
metadata:
  name: <name>
jobType: <job type>
spec:
  jobSpec:
    image: <image of workers>
    unifiedArgs:
      - <file and arguments>
    reschedulable: <true/false>
  replicaSpec:
    minReplicas: <min GPU>
    maxReplicas: <max GPU>
```

Users can see its status ```UnifiedJob.Status.UnifiedJobStatus``` to see if their job is running, waiting, or something else. 

## Examples: 

Examples can be found in [config/samples](../config/samples). 