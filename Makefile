PYTHON := python3.11
VENV := .venv
ACTIVATE := $(VENV)/bin/activate
IPYTHON := $(VENV)/bin/ipython


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

# Tasek 1 - Specific

TRAIN_CHAR_PATH ?= monkbrill_clean
AUGMENTED_CHAR_PATH ?= augmented_chars
AUGMENT_PER_CHAR ?= 100
NUM_TRAIN_SCROLLS ?= 8000
NUM_VAL_SCROLLS ?= 2000
YAML_FILE_PATH ?= src/hebrew.yaml
INPUT_SIZE ?= 1024
BATCH_SIZE ?= 32
OPTIMIZER ?= Adam
PATIENCE ?= 12
EPOCHS ?= 60
WORKERS ?= 1

task_1:
	$(IPYTHON) src/task1/scripts/data_cleaning.py
	$(IPYTHON) src/task1/scripts/noise_maps_binarizing.py
	$(IPYTHON) src/task1/scripts/create_random_scrolls.py -- --train_char_path "$(TRAIN_CHAR_PATH)" --augmented_char_path "$(AUGMENTED_CHAR_PATH)" \
		--augment_per_char $(AUGMENT_PER_CHAR) --num_train_scrolls $(NUM_TRAIN_SCROLLS) --num_val_scrolls $(NUM_VAL_SCROLLS)
	$(IPYTHON) src/task1/scripts/create_text_scrolls.py -- --train_char_path "$(TRAIN_CHAR_PATH)" --augmented_char_path "$(AUGMENTED_CHAR_PATH)" \
		--augment_per_char $(AUGMENT_PER_CHAR)
	$(IPYTHON) src/task1/scripts/line_segmentation.py
	$(IPYTHON) src/task1/scripts/train_detector.py -- --yaml_file_path "$(YAML_FILE_PATH)" --input_size $(INPUT_SIZE) \
		--batch_size $(BATCH_SIZE) --optimizer "$(OPTIMIZER)" --patience $(PATIENCE) --epochs $(EPOCHS) --workers $(WORKERS)

# Test task_1 target with smaller values
test_task_1:
	$(MAKE) task_1 \
		AUGMENT_PER_CHAR=1 \
		NUM_TRAIN_SCROLLS=50 \
		NUM_VAL_SCROLLS=20 \
		INPUT_SIZE=128 \
		EPOCHS=1

# Reset folders for task_1
reset_task_1:
	rm -rf augmented_chars dataset noise_maps/binarized
	@echo "Reset completed: removed augmented_chars, dataset, and noise_maps/binarized"