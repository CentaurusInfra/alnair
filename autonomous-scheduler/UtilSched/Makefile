all: local

local: fmt vet
	GOOS=linux GOARCH=amd64 go build  -o=bin/utilsched ./cmd/scheduler

build:  local
	sudo docker build --no-cache . -t centaurusinfra/my-scheduler:1.0.1

push:   build
	sudo docker push centaurusinfra/my-scheduler:1.0.1

# Run go fmt against code
fmt:
	go fmt ./...

# Run go vet against code
vet:
	go vet ./...

clean: fmt vet
	sudo rm -f utilsched