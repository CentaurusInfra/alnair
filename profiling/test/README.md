# Workloads Generation

## Testing with Deep Learning Training workloads

Test pod is created with [tensorflow/tensorflow:latest-gpu](https://hub.docker.com/r/tensorflow/tensorflow) image. An example training job scripts ([resnet-cifar10.py]()) is copied to the /tmp/scripts folder.

To run the workloads, first, create the test pod in your K8s cluster.

```kubectl apply -f dlt-workload.yaml``` 

Then launch the training job with the following command

```kubectl exec dlt-workload -- python3 resnet-cifar10.py```

To test more workloads, training scripts can be copied from local directory into the pod and lanuch in the same way. You can also modify the dockerfile and yaml file correspondingly.

```kubectl cp XXX.py dlt-workload:/tmp/scripts```

```kubectl exec dlt-workload -- python3 XXX.py```
