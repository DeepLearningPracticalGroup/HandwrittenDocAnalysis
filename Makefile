PYTHON := python3.11
VENV := .venv
ACTIVATE := $(VENV)/bin/activate

.PHONY: all venv install torch-cpu torch-gpu torch-help clean

all: venv install

venv:
	$(PYTHON) -m venv $(VENV)
	@echo "Virtual environment created in $(VENV)"

install: venv
	. $(ACTIVATE) && pip install --upgrade pip && pip install -r requirements.txt

torch-cpu: venv
	. $(ACTIVATE) && pip install torch torchvision
	@echo "Installed CPU version of torch and torchvision."

torch-gpu: venv
	. $(ACTIVATE) && pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
	@echo "Installed GPU (CUDA 11.8) version of torch and torchvision."

torch-help:
	@echo "Choose one of the following to install torch:"
	@echo "  make torch-cpu    # for CPU-only version"
	@echo "  make torch-gpu    # for CUDA-enabled version"

clean:
	rm -rf $(VENV)
	@echo "Virtual environment removed"
	rm -rf __pycache__
	@echo "Cache removed"
