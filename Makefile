.PHONY: help build-all run-all stop-all clean \
        build-green-agent run-green-agent stop-green-agent test-green-agent \
        build-purple-agent run-purple-agent stop-purple-agent test-purple-agent \
        test-all

# --------------------
# Load .env if present
# --------------------
ifneq (,$(wildcard .env))
    include .env
    export
endif

# --------------------
# Help
# --------------------
help:
	@echo "Available commands:"
	@echo "  make build-green-agent"
	@echo "  make run-green-agent"
	@echo "  make stop-green-agent"
	@echo "  make build-purple-agent"
	@echo "  make run-purple-agent"
	@echo "  make stop-purple-agent"
	@echo "  make build-all"
	@echo "  make run-all"
	@echo "  make stop-all"
	@echo "  make clean"
	@echo "  make test-all"

# --------------------
# Build images
# --------------------
build-green-agent:
	docker build \
		-f scenarios/cybergym/Dockerfile.green-agent \
		-t $(GREEN_AGENT_IMAGE_NAME):$(GREEN_AGENT_TAG) .

build-purple-agent:
	docker build \
		-f scenarios/cybergym/Dockerfile.purple-agent \
		-t $(PURPLE_AGENT_IMAGE_NAME):$(PURPLE_AGENT_TAG) .

build-all: build-green-agent build-purple-agent

# --------------------
# Run containers
# --------------------
run-green-agent:
	docker run -d --name cybergym-green-agent \
		-e OPENAI_API_KEY=$(OPENAI_API_KEY) \
		-e CYBERGYM_SERVER=$(CYBERGYM_DOCKER_SERVER) \
		-p $(GREEN_AGENT_PORT):$(GREEN_AGENT_PORT) \
		$(GREEN_AGENT_IMAGE_NAME):$(GREEN_AGENT_TAG)

run-purple-agent:
	docker run -d --name cybergym-purple-agent \
		-e OPENAI_API_KEY=$(OPENAI_API_KEY) \
		-p $(PURPLE_AGENT_PORT):$(PURPLE_AGENT_PORT) \
		$(PURPLE_AGENT_IMAGE_NAME):$(PURPLE_AGENT_TAG)

run-all: run-green-agent run-purple-agent

# --------------------
# Stop containers
# --------------------
stop-green-agent:
	-docker stop cybergym-green-agent
	-docker rm cybergym-green-agent

stop-purple-agent:
	-docker stop cybergym-purple-agent
	-docker rm cybergym-purple-agent

stop-all: stop-green-agent stop-purple-agent

# --------------------
# Clean images
# --------------------
clean:
	-docker rmi $(GREEN_AGENT_IMAGE_NAME):$(GREEN_AGENT_TAG)
	-docker rmi $(PURPLE_AGENT_IMAGE_NAME):$(PURPLE_AGENT_TAG)

# --------------------
# Local run (no Docker)
# --------------------
run-server-green-agent:
	uv run scenarios/cybergym/green-agent/src/server.py \
		--host 0.0.0.0 --port $(GREEN_AGENT_PORT)

run-server-purple-agent:
	uv run scenarios/cybergym/purple-agent/src/server.py \
		--host 0.0.0.0 --port $(PURPLE_AGENT_PORT)

# --------------------
# Tests
# --------------------
test-green-agent:
	uv run pytest scenarios/cybergym/green-agent/tests -v

test-purple-agent:
	uv run pytest scenarios/cybergym/purple-agent/tests -v

test-all: test-green-agent test-purple-agent
