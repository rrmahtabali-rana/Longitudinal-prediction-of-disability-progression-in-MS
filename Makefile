# =============================================================================
#  Makefile — convenience targets for the MS progression pipeline
#
#  Usage:
#      make help         show available targets
#      make all          run the full pipeline (Tasks 2-6)
#      make task2        run only Task 2 (EDA)
#      make task3        run only Task 3 (MRI stratification)
#      make task4        run Task 4 (progression labels + descriptive)
#      make task5        run only Task 5 (classification)
#      make task6        run Task 6 (survival + figure)
#      make test         run the import smoke test
#      make clean        remove all generated outputs
#      make install      install dependencies into the active environment
#      make check-data   verify the dataset is in place
# =============================================================================

PYTHON ?= python
SRC    := src

.PHONY: help all task2 task3 task4 task5 task6 test clean install check-data

help:
	@echo ""
	@echo "MS Disability Progression Prediction — make targets"
	@echo ""
	@echo "  make all          run the full pipeline (Tasks 2-6)"
	@echo "  make task2        run Task 2 (EDA)"
	@echo "  make task3        run Task 3 (MRI stratification)"
	@echo "  make task4        run Task 4 (progression + descriptive stats)"
	@echo "  make task5        run Task 5 (LR / TCN / CatBoost classification)"
	@echo "  make task6        run Task 6 (Cox / RSF survival + figure)"
	@echo "  make test         run the import smoke test"
	@echo "  make check-data   verify the dataset is accessible"
	@echo "  make install      pip install -r requirements.txt"
	@echo "  make clean        remove all generated outputs"
	@echo ""

check-data:
	@$(PYTHON) $(SRC)/paths.py

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) tests/check_imports.py

# ─── Individual tasks ──────────────────────────────────────────────────────
task2: check-data
	cd $(SRC) && $(PYTHON) task2_eda.py

task3: check-data
	cd $(SRC) && $(PYTHON) task3_stratification.py

task4: task2
	cd $(SRC) && $(PYTHON) task4_progression.py
	cd $(SRC) && $(PYTHON) task4_descriptive.py

task5: check-data
	cd $(SRC) && $(PYTHON) task5_pipeline.py

task6: task5
	cd $(SRC) && $(PYTHON) task6_survival.py
	cd $(SRC) && $(PYTHON) task6_figure.py

# ─── Full pipeline ─────────────────────────────────────────────────────────
all: task2 task3 task4 task5 task6
	@echo ""
	@echo "All tasks completed. Outputs are in ./outputs/"

# ─── Cleanup ───────────────────────────────────────────────────────────────
clean:
	rm -rf outputs/task2_outputs outputs/task3_outputs outputs/task4_outputs
	rm -rf outputs/task5_outputs outputs/task6_outputs
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned outputs/ and Python caches."
