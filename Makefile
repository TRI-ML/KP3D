PROJECT ?= kp3d
WORKSPACE ?= /workspace/$(PROJECT)
DOCKER_IMAGE ?= ${PROJECT}:latest
CPP_EXTERNALS := kp3d/externals/cpp
SHMSIZE ?= 444G
WANDB_MODE ?= run
DOCKER_OPTS := \
			--name ${PROJECT} \
			--rm -it \
			--shm-size=${SHMSIZE} \
			-e HOST_HOSTNAME= \
			-e OMP_NUM_THREADS=1 -e KMP_AFFINITY="granularity=fine,compact,1,0" \
			-e OMPI_ALLOW_RUN_AS_ROOT=1 \
			-e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
			-e NCCL_DEBUG=VERSION \
            -e DISPLAY=${DISPLAY} \
            -e XAUTHORITY \
            -e NVIDIA_DRIVER_CAPABILITIES=all \
			-v ~/.aws:/root/.aws \
			-v /root/.ssh:/root/.ssh \
			-v ~/.cache:/root/.cache \
			-v /data:/data \
			-v /mnt/fsx:/mnt/fsx \
			-v /dev/null:/dev/raw1394 \
			-v /tmp:/tmp \
			-v /tmp/.X11-unix/X0:/tmp/.X11-unix/X0 \
			-v /var/run/docker.sock:/var/run/docker.sock \
			-v ${PWD}:${WORKSPACE} \
			-w ${WORKSPACE} \
			--privileged \
			--ipc=host \
			--network=host
	
NGPUS=$(shell nvidia-smi -L | wc -l)
MPI_CMD=mpirun \
		-allow-run-as-root \
		-np ${NGPUS} \
		-H localhost:${NGPUS} \
		-x MASTER_ADDR=127.0.0.1 \
		-x MASTER_PORT=23457 \
		-x HOROVOD_TIMELINE \
		-x OMP_NUM_THREADS=1 \
		-x KMP_AFFINITY='granularity=fine,compact,1,0' \
		-bind-to none -map-by slot -x NCCL_DEBUG=INFO -x NCCL_MIN_NRINGS=4 \
		--report-bindings

.PHONY: all clean docker-build

all: clean

clean:
	find . -name "*.pyc" | xargs rm -f && \
	find . -name "__pycache__" | xargs rm -rf

$(CPP_EXTERNALS)/%: $(CPP_EXTERNALS)/%.cpp
	g++ -O3 -DNDEBUG -o $@ $@.cpp \
    $(CPP_EXTERNALS)/matrix.cpp $(CPP_EXTERNALS)/matrix.h

build-externals: kp3d/externals/cpp/evaluate_odometry 

docker-build:
	docker build \
		-f docker/Dockerfile \
		-t ${DOCKER_IMAGE} .

docker-start: docker-build
	nvidia-docker run ${DOCKER_OPTS} ${DOCKER_IMAGE} bash

docker-run: docker-build
	nvidia-docker run ${DOCKER_OPTS} ${DOCKER_IMAGE} \
		bash -c "${COMMAND}"

docker-run-mpi: docker-build
	nvidia-docker run ${DOCKER_OPTS} ${DOCKER_IMAGE} \
		bash -c "${MPI_CMD} ${COMMAND}"
