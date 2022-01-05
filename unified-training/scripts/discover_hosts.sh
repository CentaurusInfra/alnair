#!/bin/bash
workers=($(dig +short SERVICENAME.default.svc.cluster.local))
printf '%s\n' "${workers[@]/%/:1}"
