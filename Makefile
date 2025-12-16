# Makefile для новой структуры проекта (пакет main)

.PHONY: help venv pdm-env install dev test demo run clean

PYTHON = python3
PIP = pip3
VENV = .venv

help:
	@echo "Команды:"
	@echo "  make venv      - создать виртуальное окружение (venv)"
	@echo "  make install   - установить зависимости в активное окружение"
	@echo "  make run       - запустить основное приложение (GUI)"
	@echo "  make demo      - запустить демо-интерфейс"
	@echo "  make test      - запустить базовые тесты"
	@echo "  make clean     - удалить временные файлы (tmp, __pycache__)"

venv:
	$(PYTHON) -m venv $(VENV)
	@echo "Активируйте окружение командой: source $(VENV)/bin/activate"

install:
	$(PIP) install -r requirements.txt

run:
	$(PYTHON) -m main.interface.run_interface

demo:
	$(PYTHON) -m main.interface.demo_interface

test:
	$(PYTHON) -m main.tests.test_interface
	$(PYTHON) -m main.tests.test_enhancement

clean:
	rm -rf tmp
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete


