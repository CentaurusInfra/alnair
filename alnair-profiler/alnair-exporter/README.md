# Alnair exporter

## Introduction
Alnair exporter is a Prometheus exporter with custom collectors to collector metrics cross multiple layers at fine-gained, including cuda level, gpu process level, and python level.

Custom Collector

1.nvml collector

2.cpu collector

3.cuda collector

4.python collector

## Installation 
```Kubectl apply -f https://raw.githubusercontent.com/CentaurusInfra/alnair/exporter-dev/alnair-profiler/alnair-exporter/manifests/alnair-exporter.yaml```

## Prometheus scrape configuration 
