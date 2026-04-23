VENV := venv
PYTHON := $(VENV)/bin/python
CLI := $(PYTHON) -m experiment.primeintellect.cli

.PHONY: setup list-offers list-images list-pods test-provision test-inference kill-pod

setup:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -e .

list-offers:
	$(CLI) offers

list-images:
	$(CLI) images

list-pods:
	$(CLI) pods

test-provision:
	$(PYTHON) -m experiment.primeintellect --config configs/test_provision.yaml -v

test-inference:
	$(PYTHON) -m experiment.primeintellect --config configs/test_inference.yaml -v

kill-pod:
	@test -n "$(POD_ID)" || (echo "Usage: make kill-pod POD_ID=<id>" && exit 1)
	$(CLI) kill $(POD_ID)
