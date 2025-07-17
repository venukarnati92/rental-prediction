#!/bin/bash

#read from argument if not default
if [ -z "$1" ]; then
    echo "Error: host name not provided"
    exit 1  
else
    host_name="$1"
fi

export AWS_PROFILE=acg
export PREFECT_API_URL="http://${host_name}:4200/api"

#wait untill prefect server is up and running
while [ -z "$(curl -s http://$host_name:4200/api)" ]; do
    echo "Waiting for Prefect server to start..."
    sleep 5
done

python orchestration.py ${host_name} 
