# Profiler


## Image design
- Configuration using Environment Variables
  - analysis window, analysis frequency, metrics under analysis
  - prometheus service url, Pod IP, Host name, etc. 
- Query data from Prometheus server
  - deploy dcgm and profiler container in the same pod, for easy Pod IP based query
  - search localhost metrics by matching Pod ip
  - iterate all avaliablese GPU devices on the host
  - query gpu utilization and memory copy utilization metrics for analysis
- Infinit loop tracking usage every X minutes
  - reload configuration everytime to support online reconfigure
- Advanced algorithms for pattern detection and usage forecast
  - Autocorrelation based cyclic pattern detection
  - LSTM based usage forecasting
- Tag analysis results to host/node as annotation
  - grant cluster-admin role for pod service account 
  - update and log when changes happened

## Image location
Docker hub: fizzbb/profiler:v1.1
## Deploy as DaemonSet
