# Makefile для системы анализа спутниковых снимков

.PHONY: help install dev test demo run clean build dist

# Переменные
PYTHON = python3
PIP = pip3
VENV = venv
APP_NAME = satellite-analyzer

# Цвета для вывода
GREEN = \033[0;32m
YELLOW = \033[1;33m
RED = \033[0;31m
NC = \033[0m # No Color

help: ## Показать справку
	@echo "$(GREEN)Система анализа спутниковых снимков v2.0$(NC)"
	@echo "$(YELLOW)Доступные команды:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Установить зависимости
	@echo "$(YELLOW)Установка зависимостей...$(NC)"
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)✓ Зависимости установлены$(NC)"

dev: ## Установить в режиме разработки
	@echo "$(YELLOW)Установка в режиме разработки...$(NC)"
	$(PIP) install -e .
	@echo "$(GREEN)✓ Установка завершена$(NC)"

test: ## Запустить тесты
	@echo "$(YELLOW)Запуск тестов...$(NC)"
	$(PYTHON) test_interface.py
	@echo "$(GREEN)✓ Тесты завершены$(NC)"

demo: ## Запустить демонстрацию
	@echo "$(YELLOW)Запуск демонстрации...$(NC)"
	$(PYTHON) demo_interface.py

run: ## Запустить основное приложение
	@echo "$(YELLOW)Запуск основного приложения...$(NC)"
	$(PYTHON) launch_integrated.py

quick-demo: ## Быстрая демонстрация
	@echo "$(YELLOW)Быстрая демонстрация...$(NC)"
	$(PYTHON) launch_integrated.py --demo

check: ## Проверить код
	@echo "$(YELLOW)Проверка кода...$(NC)"
	@if command -v flake8 >/dev/null 2>&1; then \
		flake8 *.py --max-line-length=120; \
	else \
		echo "$(RED)flake8 не установлен, пропускаем проверку$(NC)"; \
	fi

format: ## Форматировать код
	@echo "$(YELLOW)Форматирование кода...$(NC)"
	@if command -v black >/dev/null 2>&1; then \
		black *.py --line-length=120; \
	else \
		echo "$(RED)black не установлен, пропускаем форматирование$(NC)"; \
	fi

build: ## Собрать пакет
	@echo "$(YELLOW)Сборка пакета...$(NC)"
	$(PYTHON) setup.py sdist bdist_wheel
	@echo "$(GREEN)✓ Пакет собран$(NC)"

dist: ## Создать дистрибутив
	@echo "$(YELLOW)Создание дистрибутива...$(NC)"
	$(PYTHON) -m build
	@echo "$(GREEN)✓ Дистрибутив создан$(NC)"

clean: ## Очистить временные файлы
	@echo "$(YELLOW)Очистка временных файлов...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "$(GREEN)✓ Очистка завершена$(NC)"

venv: ## Создать виртуальное окружение
	@echo "$(YELLOW)Создание виртуального окружения...$(NC)"
	$(PYTHON) -m venv $(VENV)
	@echo "$(GREEN)✓ Виртуальное окружение создано$(NC)"
	@echo "$(YELLOW)Активируйте его командой: source $(VENV)/bin/activate$(NC)"

activate: ## Показать команду активации venv
	@echo "$(YELLOW)Для активации виртуального окружения выполните:$(NC)"
	@echo "source $(VENV)/bin/activate"

setup: venv install ## Полная настройка проекта
	@echo "$(GREEN)✓ Проект настроен и готов к работе$(NC)"

all: clean install test ## Полная проверка проекта
	@echo "$(GREEN)✓ Все проверки пройдены$(NC)"

# Специальные команды для разработки
dev-install: ## Установить dev зависимости
	@echo "$(YELLOW)Установка dev зависимостей...$(NC)"
	$(PIP) install flake8 black pytest mypy
	@echo "$(GREEN)✓ Dev зависимости установлены$(NC)"

lint: ## Проверить код линтером
	@echo "$(YELLOW)Проверка кода...$(NC)"
	@if command -v mypy >/dev/null 2>&1; then \
		mypy *.py --ignore-missing-imports; \
	else \
		echo "$(RED)mypy не установлен$(NC)"; \
	fi

# Команды для упаковки
package: clean build ## Создать пакет
	@echo "$(GREEN)✓ Пакет готов$(NC)"

upload: dist ## Загрузить в PyPI (требует настройки)
	@echo "$(YELLOW)Загрузка в PyPI...$(NC)"
	twine upload dist/*
	@echo "$(GREEN)✓ Загружено$(NC)"

# Информационные команды
info: ## Показать информацию о проекте
	@echo "$(GREEN)Информация о проекте:$(NC)"
	@echo "  Название: Система анализа спутниковых снимков"
	@echo "  Версия: 2.0.0"
	@echo "  Python: $(shell $(PYTHON) --version)"
	@echo "  Pip: $(shell $(PIP) --version)"
	@echo "  Окружение: $(shell which $(PYTHON))"

status: ## Показать статус проекта
	@echo "$(GREEN)Статус проекта:$(NC)"
	@echo "  Файлы Python: $(shell find . -name '*.py' | wc -l)"
	@echo "  Строки кода: $(shell find . -name '*.py' -exec wc -l {} + | tail -1)"
	@echo "  Размер проекта: $(shell du -sh . | cut -f1)"

# По умолчанию показываем справку
.DEFAULT_GOAL := help
