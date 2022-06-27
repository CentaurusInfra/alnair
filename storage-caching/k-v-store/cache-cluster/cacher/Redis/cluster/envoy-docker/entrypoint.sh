#!/bin/bash

cat /tmp/envoy/envoy.yaml | envsubst \$REDIS_PWD,\$REDIS_PROXY_USERNAME,\$REDIS_PROXY_PWD > /etc/envoy/envoy.yaml
envoy -c /etc/envoy/envoy.yaml -l info
