.PHONY: requirements data lint environment clean_environment clean test dev_requirements

# _________________________ GLOBALS _________________________

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = deepnn
PYTHON_INTERPRETER = python
PACKAGE_MANAGER = pip
PYTHON_VERSION = 3

# _________________________ COMMANDS _________________________

# In case a command need args
#args = `arg="$(filter-out $@,$(MAKECMDGOALS))" && echo $${arg:-${1}}`

# to use args
# $(call args, <default_value>)

## Install Python Dependencies for development
requirements:
	$(PYTHON_INTERPRETER) -m pip install --no-cache-dir -U pip setuptools wheel build
	$(PYTHON_INTERPRETER) -m pip install --no-cache-dir -U -e .[build,test,dev]

## Lint using flake8
lint:
	python -m tox -p -e flake8,mypy

## Test using pytest
test:
	python -m pytest test

## Test using pytest and coverage
cov-test:
	python -m tox -e py310

## Create python virtual environment
environment:
ifneq (,$(wildcard ./venv))
	@echo The virtual environment has been already created;
else
ifeq ($(PACKAGE_MANAGER),"conda")
	@echo ">>> Detected conda, creating conda environment."
	$(PACKAGE_MANAGER) create --name $(PROJECT_NAME) python=$(PYTHON_VERSION)
	@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	@echo Creating venv venv;
	$(PYTHON_INTERPRETER) -m pip install -U pip virtualenv;
	$(PYTHON_INTERPRETER) -m virtualenv venv;
endif
endif

## Clean virtual environment
clean_environment:
ifneq (,$(wildcard ./venv))
	deactivate || echo "Already venv deactivated";
	rm -r venv;
else
	@echo It has been cleaned already
endif

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	python -m tox -e clean


# _________________________ Self Documenting Commands _________________________

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
