PYTHON_VERSION := python3.11
VENV_DIR := .venv

check-brew:
	@command -v brew >/dev/null 2>&1 || (\
		echo "❌ Homebrew is not installed."; \
		echo "➡️  Please install it using:"; \
		echo '/bin/bash -c "$$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'; \
		exit 1)

check-python:
	@command -v $(PYTHON_VERSION) >/dev/null 2>&1 || (\
		echo "❌ $(PYTHON_VERSION) not found."; \
		echo "➡️  Installing with Homebrew..."; \
		brew install python@3.11)

setup: check-brew check-python
	@echo "Creating virtual environment with $(PYTHON_VERSION)..."
	$(PYTHON_VERSION) -m venv $(VENV_DIR)
	@echo "Activating virtual environment and installing dependencies..."
	. $(VENV_DIR)/bin/activate && pip install --upgrade pip && pip install -r requirements.txt
	@echo "✅ Setup complete. Activate with: source $(VENV_DIR)/bin/activate"

run:
	@. $(VENV_DIR)/bin/activate && python your_code.py

clean:
	rm -rf $(VENV_DIR)
