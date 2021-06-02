# Release Summary

This release is the first release of the project. It includes the following new components:

- Profiler (Daemon Set)
- Elastic training operator


# Key Features and Improvements:
- Profiler
  - GPU metrics collection
    - Take advantage of Nvidia monitoring toolkit DCGM-exporter, device-level metrics, e.g. GPU and Memory utilization are collected every second.
    - The GPU metrics exported from DCGM is scraped by Prometheus. Prometheus auto discovers the pods with scraping annoations.
  - Deep learning training job (DLT) identification
    - Considering the cyclic pattern of memory utilization in DLT jobs, an autocorrelation based cyclic pattern detection algorithm is implemented to detect DLT job, once DLT job is detected, the max memory utilization is predicted based on the past usage.
    - Analytical results including job type and predicted memotry utilization are continuously patched to every GPU node as annotations.
