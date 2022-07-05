/**
 * Copyright 2022 Steven Wang, Futurewei Inc
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
 package exporter

 import "github.com/prometheus/client_golang/prometheus"
 
 // AddMetrics - Add's all of the metrics to a map of strings, returns the map.
 func AddMetrics() map[string]*prometheus.Desc {
 
	 APIMetrics := make(map[string]*prometheus.Desc)
 
 
 
	 APIMetrics["BurstSize"] = prometheus.NewDesc(
		 prometheus.BuildFQName("GPU", "pod", "burst"),
		 "The elapse time (milliseconds)at which the current kernel runs on GPU.",
		 []string{"pod", "gpu_uuid"}, nil,
	 )	
	 APIMetrics["Overuse"] = prometheus.NewDesc(
		 prometheus.BuildFQName("GPU", "pod", "overuse"),
		 "The elapse time (milliseconds)at which the current kernel runs overtime on GPU.",
		 []string{"pod", "gpu_uuid"}, nil,
	 )	
	 APIMetrics["MemH2D"] = prometheus.NewDesc(
		 prometheus.BuildFQName("GPU", "pod", "H2D"),
		 "The elapse time (milliseconds)at which the current pod memory copy (H2D) on GPU.",
		 []string{"pod", "gpu_uuid"}, nil,
	 )	
	 APIMetrics["MemD2H"] = prometheus.NewDesc(
		 prometheus.BuildFQName("GPU", "pod", "D2H"),
		 "The elapse time (milliseconds)at which the current pod memory copy(D2H) on GPU.",
		 []string{"pod", "gpu_uuid"}, nil,
	 )	
	 // APIMetrics["MemRem"] = prometheus.NewDesc(
	 // 	prometheus.BuildFQName("GPU", "pod", "MemRemain"),
	 // 	"How much memory is left on GPU.",
	 // 	[]string{"pod", "gpu_uuid"}, nil,
	 // )	
	 APIMetrics["TimeRem"] = prometheus.NewDesc(
		 prometheus.BuildFQName("GPU", "pod", "TimeRemain"),
		 "How much time is left for the pod to run on the GPU.",
		 []string{"pod", "gpu_uuid"}, nil,
	 )	
	 APIMetrics["MemSize"] = prometheus.NewDesc(
		 prometheus.BuildFQName("GPU", "pod", "MemorySize"),
		 "The maxium memory is required by the pod.",
		 []string{"pod", "gpu_uuid"}, nil,
	 )	
	 APIMetrics["MemUsed"] = prometheus.NewDesc(
		 prometheus.BuildFQName("GPU", "pod", "MemoryUsed"),
		 "How much memory is used by the pod.",
		 []string{"pod", "gpu_uuid"}, nil,
	 )	
	 return APIMetrics
 }
 
 
 // processGPUmetrics - processes the response GPU metrics using it as a source
 func (e *Exporter) processGPUMetrics( data *GPUMetrics, ch chan<- prometheus.Metric, podname string, uuid string) error {
 
	 // Set Rate limit stats
	 // "{\"Ts\": %ld, \"Bs\": %d, \"Ou\": %d, \"Rm\": %d, \"Mm\": %d}",
	 ch <- prometheus.MustNewConstMetric(e.APIMetrics["BurstSize"], prometheus.GaugeValue, float64(data.Bs), podname, uuid)
	 ch <- prometheus.MustNewConstMetric(e.APIMetrics["Overuse"], prometheus.GaugeValue, float64(data.Ou), podname, uuid)
	 ch <- prometheus.MustNewConstMetric(e.APIMetrics["MemH2D"], prometheus.GaugeValue, float64(data.Hd), podname, uuid)
	 ch <- prometheus.MustNewConstMetric(e.APIMetrics["MemD2H"], prometheus.GaugeValue, float64(data.Dh), podname, uuid)
	 ch <- prometheus.MustNewConstMetric(e.APIMetrics["TimeRem"], prometheus.GaugeValue, float64(data.Rm), podname, uuid)
	 ch <- prometheus.MustNewConstMetric(e.APIMetrics["MemSize"], prometheus.GaugeValue, float64(data.Mm), podname, uuid)
	 ch <- prometheus.MustNewConstMetric(e.APIMetrics["MemUsed"], prometheus.GaugeValue, float64(data.Um), podname, uuid)
 
	 return nil
 }
 