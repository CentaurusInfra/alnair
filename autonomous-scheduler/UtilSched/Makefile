all: local

local: fmt vet
	GOOS=linux GOARCH=amd64 go build  -o=bin/utilsched ./cmd/scheduler

build:  local
	docker build --no-cache . -t https://hub.docker.com/r/centaurusinfra/my-scheduler/utilsched:1.0.1

push:   build
	docker push https://hub.docker.com/r/centaurusinfra/my-scheduler/utilsched:1.0.1

# Run go fmt against code
fmt:
	go fmt ./...

# Run go vet against code
vet:
	go vet ./...

clean: fmt vet
	sudo rm -f utilsched