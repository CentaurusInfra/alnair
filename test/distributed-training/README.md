## Pytorch DDP
mnist-distributed.py is an pytorch DDP example.

**Make sure to change the [MASTER_IP](https://github.com/CentaurusInfra/alnair/blob/main/test/distributed-training/mnist-distributed.py#L25) to your own host's IP.**
### Launch
### 1. single node multiple GPUs
```
python mnist-distributed.py -n 1 -g 8 -i 0
```
**NOTES**: After a couple of successful run with different gpu counts(>1), the program failed at  ```-g 1```. 

Complains about **some cuda functions before calling NumCudaDevices() that might have already set an error? Error 101: invalid device ordinal (function operator())**

** torch.cuda.is_available() returned false. nvidia-smi shows 7 GPUs instead of 8. **

** Reboot server gpu counts back to 8, and rerun the commands, error did not reproduce. **

### 2. multiple(two) nodes multiple GPUs
on each node launch separately 
```
python mnist-distributed.py -n 2 -g 8,2 -i 0
python mnist-distributed.py -n 2 -g 8,2 -i 1
```
In the above code, the first worker (master node) has 8 GPUs and the second worker machine has 2 GPUs.

### Cross node network throughput measurement
1. use tcp dump, assume use port 8765

```tcpdump -i any port 8765 |pv -bert > /dev/null```

Expect Results
```
0.00 B 0:00:02 [0.00 B/s]o: 0.00 B 0:00:01 [0.00 B/s]
tcpdump: verbose output suppressed, use -v or -vv for full protocol decode
listening on any, link-type LINUX_SLL (Linux cooked v1), capture size 262144 bytes
20.0KiB 0:00:17 [4.01KiB/s]
```
### Catch
1. env MASTER_IP needs to be localhost IP, when use in the single node multiple card mode, so when you test out the script on different nodes, remember to change the IP address. Otherwise the program will keep waiting to join the MASTER IP's process.
2. If you multiple nodes, make sure use full amount of gpu. If both nodes have 8 cards, you only launch 4 gpu on each nodes, the rank/gpu device setup may mess up. currently only tested 2 nodes, and each nodes have 8 cards scenarios. If use partial GPUs from each node, may need to set CUDA_VISIBLE_DEVICE in the env.

### troubleshooting
#### 1. connectin refused
```
Connect [127.0.1.1]:[a port]: Connection refused
```
**solution**

```
export GLOO_SOCKET_IFNAME=eth1  ## change eth1 to the target interface
```
#### 2. Error: "Address is already in use" 
Previous multi-process may be lauched and not cleaned up properly, resulting the master port is in use

Find the pid of the previous multi process, and kill it 
```
ps -aux | grep python
kill -9 <pid>
```
## Horovd
