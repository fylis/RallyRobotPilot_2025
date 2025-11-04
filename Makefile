# Targets to manage the headless service and its image
SERVICE := headless
IMAGE_NAME := 2025-rally-robot
DOCKERFILE := Dockerfile.headless
CONTEXT := .

.PHONY: build build-direct up run rebuild down clean

# Use docker compose to build and tag the image as defined in docker-compose.yaml
build:
	docker compose build $(SERVICE)

up:
	docker compose up $(SERVICE)

run:
	docker compose run --rm $(SERVICE) python3 $(file)

rebuild: down clean build

down:
	docker compose down

clean:
	@if docker image inspect $(IMAGE_NAME) > /dev/null 2>&1; then \
		echo "Removing image $(IMAGE_NAME)..."; \
		docker image rm -f $(IMAGE_NAME); \
	else \
		echo "Image $(IMAGE_NAME) not found. Skipping."; \
	fi
