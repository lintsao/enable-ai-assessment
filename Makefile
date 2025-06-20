.PHONY: help run format lint clean

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

run: ## Run the Streamlit app
	streamlit run src/mouth_open_detector_streamlit_combined.py

format: ## Format code with black and isort
	black src/
	isort src/

lint: ## Run linting with flake8
	flake8 src/

check: format lint ## Run formatting and linting

clean: ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 