# User Usage 

## Docker Image 
The docker image the user provides is standard, although the user should ensure ```python``` will refer to the desired Python location. 

## Checkpoint Files 
If the user is using checkpoints, there should be the flag ```--checkpoint-dir``` as an argument for the script. This will be system-designated and the user should not specify it. 

## YAML Usage

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

Users can see its status ```UnifiedJob.Status.UnifiedJobStatus```. 

## Logs 

Logs during runtime can be seen through the standard ```kubectl logs [...]```. After runtime, the pods are all currently deleted; in the future, these logs may all be concatenated into another place. 