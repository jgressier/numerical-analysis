# Define variables
PYTHON = python3
VENV_DIR = venv
REQ_FILE = requirements.txt
BOOK_SRC = book  # Path to Jupyter Book source (default current directory)
BOOK_BUILD_DIR = book/_build

# Default target
all: venv install build

# Create virtual environment
venv:
	$(PYTHON) -m venv $(VENV_DIR)

# Install dependencies
install: venv
	$(VENV_DIR)/bin/pip install -r $(REQ_FILE)

# Build Jupyter Book
build: install
	$(VENV_DIR)/bin/jupyter-book build $(BOOK_SRC)

# Clean build files
clean:
	rm -rf $(BOOK_BUILD_DIR)

# Clean virtual environment
clean-venv:
	rm -rf $(VENV_DIR)

# Full clean
clean-all: clean clean-venv