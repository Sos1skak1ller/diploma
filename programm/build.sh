#!/bin/bash

# Скрипт сборки для системы анализа спутниковых снимков
# Версия: 2.0.0

set -e  # Остановка при ошибке

# Цвета для вывода
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Функция для вывода сообщений
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✓${NC} $1"
}

warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

error() {
    echo -e "${RED}✗${NC} $1"
}

# Заголовок
echo -e "${GREEN}"
echo "=============================================="
echo "  Система анализа спутниковых снимков v2.0"
echo "  Скрипт сборки и развертывания"
echo "=============================================="
echo -e "${NC}"

# Проверка Python
log "Проверка Python..."
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    PIP_CMD="pip3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
    PIP_CMD="pip"
else
    error "Python не найден! Установите Python 3.7+"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
log "Найден Python версии: $PYTHON_VERSION"

# Проверка версии Python
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 7 ]); then
    error "Требуется Python 3.7 или выше. Текущая версия: $PYTHON_VERSION"
    exit 1
fi

success "Python версия подходит"

# Проверка виртуального окружения
log "Проверка виртуального окружения..."
if [ -d "venv" ]; then
    success "Виртуальное окружение найдено"
    source venv/bin/activate
    log "Виртуальное окружение активировано"
else
    warning "Виртуальное окружение не найдено, создаем..."
    $PYTHON_CMD -m venv venv
    source venv/bin/activate
    success "Виртуальное окружение создано и активировано"
fi

# Обновление pip
log "Обновление pip..."
$PIP_CMD install --upgrade pip
success "pip обновлен"

# Установка зависимостей
log "Установка зависимостей..."
if [ -f "requirements.txt" ]; then
    $PIP_CMD install -r requirements.txt
    success "Зависимости установлены"
else
    error "Файл requirements.txt не найден!"
    exit 1
fi

# Проверка установки PyQt5
log "Проверка PyQt5..."
if $PYTHON_CMD -c "import PyQt5" 2>/dev/null; then
    success "PyQt5 установлен"
else
    error "PyQt5 не установлен! Установка..."
    $PIP_CMD install PyQt5
    success "PyQt5 установлен"
fi

# Тестирование
log "Запуск тестов..."
if [ -f "test_interface.py" ]; then
    if $PYTHON_CMD test_interface.py; then
        success "Тесты пройдены"
    else
        warning "Тесты не пройдены, но продолжаем сборку"
    fi
else
    warning "Файл тестов не найден"
fi

# Создание пакета
log "Создание пакета..."
if [ -f "setup.py" ]; then
    $PYTHON_CMD setup.py sdist bdist_wheel
    success "Пакет создан"
else
    warning "setup.py не найден, пропускаем создание пакета"
fi

# Создание исполняемых файлов
log "Создание исполняемых файлов..."
mkdir -p bin

# Создание скрипта запуска
cat > bin/run_satellite_analyzer << 'EOF'
#!/bin/bash
cd "$(dirname "$0")/.."
source venv/bin/activate
python launch_integrated.py "$@"
EOF

chmod +x bin/run_satellite_analyzer
success "Скрипт запуска создан: bin/run_satellite_analyzer"

# Создание скрипта демо
cat > bin/run_demo << 'EOF'
#!/bin/bash
cd "$(dirname "$0")/.."
source venv/bin/activate
python demo_interface.py "$@"
EOF

chmod +x bin/run_demo
success "Скрипт демо создан: bin/run_demo"

# Создание скрипта тестов
cat > bin/run_tests << 'EOF'
#!/bin/bash
cd "$(dirname "$0")/.."
source venv/bin/activate
python test_interface.py "$@"
EOF

chmod +x bin/run_tests
success "Скрипт тестов создан: bin/run_tests"

# Создание desktop файла для Linux
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    log "Создание desktop файла..."
    cat > satellite-analyzer.desktop << EOF
[Desktop Entry]
Version=2.0
Type=Application
Name=Satellite Image Analyzer
Comment=Система анализа спутниковых снимков
Exec=$(pwd)/bin/run_satellite_analyzer
Icon=$(pwd)/icon.png
Terminal=false
Categories=Graphics;Science;Education;
EOF
    success "Desktop файл создан"
fi

# Создание README для сборки
log "Создание README для сборки..."
cat > BUILD_README.md << 'EOF'
# Сборка завершена!

## Быстрый запуск

### Основное приложение
```bash
./bin/run_satellite_analyzer
```

### Демонстрация
```bash
./bin/run_demo
```

### Тесты
```bash
./bin/run_tests
```

## Альтернативные способы запуска

### Через Python
```bash
source venv/bin/activate
python launch_integrated.py
```

### Демо
```bash
source venv/bin/activate
python demo_interface.py
```

## Структура проекта

- `bin/` - Исполняемые скрипты
- `venv/` - Виртуальное окружение
- `dist/` - Собранные пакеты
- `*.py` - Исходный код

## Требования

- Python 3.7+
- PyQt5
- Виртуальное окружение активировано

## Поддержка

При возникновении проблем проверьте:
1. Активировано ли виртуальное окружение
2. Установлены ли все зависимости
3. Корректность путей к файлам
EOF

success "README для сборки создан"

# Финальная информация
echo -e "${GREEN}"
echo "=============================================="
echo "  СБОРКА ЗАВЕРШЕНА УСПЕШНО!"
echo "=============================================="
echo -e "${NC}"

echo -e "${YELLOW}Доступные команды:${NC}"
echo "  ./bin/run_satellite_analyzer  - Запуск основного приложения"
echo "  ./bin/run_demo               - Демонстрация"
echo "  ./bin/run_tests              - Тесты"
echo ""
echo -e "${YELLOW}Альтернативно:${NC}"
echo "  make run                     - Запуск через Makefile"
echo "  make demo                    - Демо через Makefile"
echo "  make test                    - Тесты через Makefile"
echo ""
echo -e "${GREEN}Готово к использованию!${NC}"
