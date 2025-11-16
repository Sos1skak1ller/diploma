#!/usr/bin/env python3
"""
Простой тест интерфейса (перенесён в пакет main.tests)
"""

import sys
from PyQt5.QtWidgets import QApplication


def test_basic_imports():
    """Тест базовых импортов"""
    print("Тестирование импортов...")

    try:
        from PyQt5.QtWidgets import QMainWindow, QTabWidget  # noqa: F401
        print("✓ PyQt5 импортирован успешно")
    except ImportError as e:
        print(f"✗ Ошибка импорта PyQt5: {e}")
        return False

    try:
        from main.interface.main_interface import MainInterface  # noqa: F401
        print("✓ MainInterface импортирован успешно")
    except ImportError as e:
        print(f"✗ Ошибка импорта MainInterface: {e}")
        return False

    try:
        from main.interface.new_controller import NewController  # noqa: F401
        print("✓ NewController импортирован успешно")
    except ImportError as e:
        print(f"✗ Ошибка импорта NewController: {e}")
        return False

    return True


def test_interface_creation():
    """Тест создания интерфейса"""
    print("\nТестирование создания интерфейса...")

    app = QApplication(sys.argv)

    try:
        from main.interface.main_interface import MainInterface

        window = MainInterface()

        # Проверяем основные компоненты
        assert hasattr(window, "tab_widget"), "Отсутствует tab_widget"
        assert window.tab_widget.count() == 3, f"Неверное количество вкладок: {window.tab_widget.count()}"

        # Проверяем названия вкладок
        tab_names = [window.tab_widget.tabText(i) for i in range(window.tab_widget.count())]
        expected_tabs = [
            "Главная - Детекция объектов",
            "Зоны интересов",
            "Улучшение качества снимков",
        ]

        for expected in expected_tabs:
            assert expected in tab_names, f"Отсутствует вкладка: {expected}"

        print("✓ Интерфейс создан успешно")
        print(f"✓ Найдено вкладок: {window.tab_widget.count()}")
        print(f"✓ Названия вкладок: {tab_names}")

        return True

    except Exception as e:
        print(f"✗ Ошибка создания интерфейса: {e}")
        return False


def test_controller_creation():
    """Тест создания контроллера"""
    print("\nТестирование создания контроллера...")

    try:
        from main.interface.main_interface import MainInterface
        from main.interface.new_controller import NewController

        app = QApplication.instance()
        if not app:
            app = QApplication(sys.argv)

        window = MainInterface()
        controller = NewController(window)

        # Проверяем основные атрибуты контроллера
        assert hasattr(controller, "view"), "Отсутствует view в контроллере"
        assert hasattr(controller, "current_workers"), "Отсутствует current_workers в контроллере"

        print("✓ Контроллер создан успешно")
        print("✓ Связи между view и controller установлены")

        return True

    except Exception as e:
        print(f"✗ Ошибка создания контроллера: {e}")
        return False


def main():
    """Главная функция тестирования"""

    print("=" * 50)
    print("Тестирование нового интерфейса (пакет main)")
    print("=" * 50)

    # Тест 1: Импорты
    if not test_basic_imports():
        print("\n❌ Тест импортов не пройден")
        return 1

    # Тест 2: Создание интерфейса
    if not test_interface_creation():
        print("\n❌ Тест создания интерфейса не пройден")
        return 1

    # Тест 3: Создание контроллера
    if not test_controller_creation():
        print("\n❌ Тест создания контроллера не пройден")
        return 1

    print("\n" + "=" * 50)
    print("✅ Все тесты пройдены успешно!")
    print("Интерфейс готов к использованию")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(main())


