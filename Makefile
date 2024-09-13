VENV = venv

# Target to create the virtual environment
.PHONY: install
install:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt

# Target to delete the virtual environment
.PHONY: clean
clean:
	rm -rf $(VENV)

# Target to run the main application
.PHONY: run
run:
	$(VENV)/bin/python lstm(forecasting).py
