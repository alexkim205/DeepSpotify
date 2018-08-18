# Make rules for Docker images

MY_DOCKER=alexkim205/keras_music
VERSION=0.0.1

PORT=8888
SRC?=$(shell dirname `pwd`)

# HELP
# This will output the help for each task
.PHONY: help

help: ## This help.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.DEFAULT_GOAL := help

# DOCKER TASKS
# Build the container

#https://github.com/gw0/docker-debian-cuda
build: ## Build Keras + Music Docker
	docker build -t $(MY_DOCKER):$(VERSION) -t $(MY_DOCKER):latest -f ./Dockerfile .

notebook: ## Run Jupyter on CPU Docker
	docker run -it --rm -p $(PORT):8888 -v $(SRC):/srv $(MY_DOCKER)


# CONDA TASKS
envexport: ## Export music Conda environment to music.yaml
	conda env export > music.yaml

envimport: ## Import music Conda environment from music.yaml
	conda env create -f music.yaml
