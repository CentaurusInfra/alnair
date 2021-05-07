# Fine-grained GPU sharing

## Existing sharing strategies
- Time slicing, each application gets exclusive access for a period before access is granted to another application (context switching overhead)
- Multi-process service (MPS) combine applications, memory partition (processes interference)
- GPU stream for parallel inference, with TensorRT server
- Low-level CUDA API modification, to achieve better isolation and QoS
- Multi-Instance GPU (MIG), with latest A100 GPU

