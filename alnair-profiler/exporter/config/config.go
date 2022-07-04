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
 package config

 import (
	 "io/ioutil"
	 "net/url"
	 "path"
	 "strings"
 
	 log "github.com/sirupsen/logrus"
 
	 "os"
 
	 cfg "github.com/pint1022/go-common/config"
 )
 
 // Config struct holds all of the runtime confgiguration for the application
 type Config struct {
	 *cfg.BaseConfig
	 sampleRate    string
	 alnrIP   string
	 alnrPort string
 }
 
 // Init populates the Config struct based on environmental runtime configuration
 func Init() Config {
 
	 listenPort := cfg.GetEnv("LISTEN_PORT", "9171")
	 os.Setenv("LISTEN_PORT", listenPort)
	 ac := cfg.Init()
 
	 appConfig := Config{
		 &ac,
		 "1000",
		 "alnr-exporter.kube-system.svc",
		 "60018",
	 }
 
	 sampleRate := os.Getenv("SAMPLE_RATE");
	 if (sampleRate != "") {
	   appConfig.SetSampleRate(sampleRate);
	 }
   
	 alnrIP := os.Getenv("EXPORTER_IP");
	 if (alnrIP != "") {
	   appConfig.SetAlnrIP(alnrIP);
	 }
   
	 alnrPort := os.Getenv("ALNR_PORT");
	 if (alnrIP != "") {
	   appConfig.SetAlnrPort(alnrPort);
	 }  
 
	 return appConfig
 }
 
 // SetSampleRate accepts a string of sampling rate
 func (c *Config) SetSampleRate(token string) {
	 c.sampleRate = token
 }
 
 // Returns the sample rate for sampling
 func (c *Config) SampleRate() string {
	 return c.sampleRate
 }
 
 // SetSchedulerIP accepts a string of scheduler IP
 func (c *Config) SetAlnrIP(token string) {
	 c.alnrIP = token
 }
 
 // Returns scheduler IP address
 func (c *Config) AlnrIP() string {
	 return c.alnrIP
 }
 
 // SetSchedulerPort accepts a string of scheduler Port
 func (c *Config) SetAlnrPort(token string) {
	 c.alnrPort = token
 }
 
 // Returns scheduler port
 func (c *Config) AlnrPort() string {
	 return c.alnrPort
 }
 