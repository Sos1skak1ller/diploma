# Makefile для новой структуры проекта (пакет main)

.PHONY: help venv pdm-env install dev test demo run clean \
        server-up server-down server-logs server-build

PYTHON = python3
PIP = pip3
VENV = .venv
SERVER_DIR = image_server
DOCKER_COMPOSE = docker compose

help:
	@echo "Команды:"
	@echo "  make venv         - создать виртуальное окружение (venv)"
	@echo "  make install      - установить зависимости в активное окружение"
	@echo "  make run          - запустить основное приложение (GUI)"
	@echo "  make demo         - запустить демо-интерфейс"
	@echo "  make test         - запустить базовые тесты"
	@echo "  make clean        - удалить временные файлы (tmp, __pycache__)"
	@echo "  make server-up    - поднять локальный демо-сервер снимков (Docker)"
	@echo "  make server-down  - остановить демо-сервер снимков"
	@echo "  make server-logs  - читать логи демо-сервера"
	@echo "  make server-build - пересобрать образ демо-сервера"

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

server-up:
	cd $(SERVER_DIR) && $(DOCKER_COMPOSE) up -d --build
	@echo "Демо-сервер: http://localhost:8000  (документация: /docs)"

server-down:
	cd $(SERVER_DIR) && $(DOCKER_COMPOSE) down

server-logs:
	cd $(SERVER_DIR) && $(DOCKER_COMPOSE) logs -f

server-build:
	cd $(SERVER_DIR) && $(DOCKER_COMPOSE) build


