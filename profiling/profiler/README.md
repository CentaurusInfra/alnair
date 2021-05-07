# Continuous GPU usage profiling


## Image design
- Configuration by Environment Variables
  - analysis window, analysis frequency, metrics under analysis
  - prometheus service url, host ip, 
- Query data from prometheus server
  - search localhost metrics by matching host ip
  - iterate all avaliablese GPU devices on the host
  - query gpu utilization and memory copy utilization metrics for analysis
- Infinit loop tracking usage every X minutes
  - reload configuration everytime to support online reconfigure
- Advanced seasonality detection and usage forecasting algorithms
  - Autocorrelation based seasonality detection
  - LSTM (or transformer) based usage forecasting
- Tag analysis results to host/node as annotation
  - grant cluster-admin role for pod service account 
  - update and log when changes happened

## Image location
Docker hub: fizzbb/profiler:v1
## Deploy as DaemonSet
