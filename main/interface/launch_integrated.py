#!/usr/bin/env python3
"""
Главный запускающий файл для интегрированной системы (перенесён в пакет main.interface).
"""

import sys
from PyQt5.QtWidgets import QApplication, QMessageBox, QSplashScreen
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QFont


def show_splash_screen():
    """Показывает splash screen при загрузке"""
    
    # Создаем простое изображение для splash screen
    pixmap = QPixmap(400, 300)
    pixmap.fill(Qt.darkGray)
    
    splash = QSplashScreen(pixmap)
    splash.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.SplashScreen)
    
    # Добавляем текст
    font = QFont("Arial", 16, QFont.Bold)
    splash.setFont(font)
    
    splash.show()
    splash.showMessage(
        "Загрузка системы анализа\nспутниковых снимков...",
        Qt.AlignCenter | Qt.AlignBottom,
        Qt.white
    )
    
    return splash


def check_dependencies():
    """Проверяет зависимости"""
    
    try:
        from PyQt5 import QtCore
        print(f"PyQt5 версия: {QtCore.PYQT_VERSION_STR}")
    except Exception as e:
        print(f"Ошибка: PyQt5 не установлен или некорректен: {e}")
        return False
        
    return True


def main():
    """Главная функция"""
    
    print("=" * 60)
    print("Система анализа спутниковых снимков v2.0")
    print("=" * 60)
    
    # Проверяем зависимости
    if not check_dependencies():
        print("Ошибка: Не все зависимости установлены")
        return 1
        
    # Создаем приложение Qt
    app = QApplication(sys.argv)
    app.setApplicationName("Система анализа спутниковых снимков")
    app.setApplicationVersion("2.0")
    
    # Показываем splash screen
    splash = show_splash_screen()
    app.processEvents()
    
    try:
        # Импортируем основные модули
        from .main_interface import MainInterface
        from .new_controller import NewController
        from .integration_bridge import IntegrationBridge
        
        # Создаем главное окно
        splash.showMessage("Создание интерфейса...", Qt.AlignCenter | Qt.AlignBottom, Qt.white)
        app.processEvents()
        
        window = MainInterface()
        
        # Создаем контроллер
        splash.showMessage("Инициализация контроллера...", Qt.AlignCenter | Qt.AlignBottom, Qt.white)
        app.processEvents()
        
        controller = NewController(window)
        try:
            window.controller = controller
        except Exception:
            pass
        
        # Создаем мост интеграции (пока без legacy-модели)
        splash.showMessage("Настройка интеграции...", Qt.AlignCenter | Qt.AlignBottom, Qt.white)
        app.processEvents()
        
        bridge = IntegrationBridge(window)
        _ = bridge  # чтобы не было предупреждения о неиспользуемой переменной
        
        # Завершаем splash screen
        splash.showMessage("Загрузка завершена!", Qt.AlignCenter | Qt.AlignBottom, Qt.green)
        app.processEvents()
        
        # Ждем немного перед закрытием splash screen
        QTimer.singleShot(1000, splash.close)
        
        # Показываем главное окно
        QTimer.singleShot(1200, window.show)
        
        print("Приложение успешно запущено")
        print("Доступные вкладки:")
        print("  1. Главная - Детекция объектов")
        print("  2. Зоны интересов")
        print("  3. Улучшение качества снимков")
        print("=" * 60)
        
        # Запускаем главный цикл приложения
        return app.exec_()
        
    except ImportError as e:
        splash.close()
        QMessageBox.critical(
            None,
            "Ошибка импорта",
            f"Не удалось импортировать необходимые модули:\n{str(e)}\n\n"
            "Убедитесь, что пакет main установлен корректно."
        )
        return 1
        
    except Exception as e:
        splash.close()
        QMessageBox.critical(
            None,
            "Ошибка запуска",
            f"Произошла ошибка при запуске приложения:\n{str(e)}"
        )
        print(f"Критическая ошибка: {e}")
        return 1


def launch_demo():
    """Запуск демонстрационной версии"""
    
    print("Запуск демонстрационной версии...")
    
    app = QApplication(sys.argv)
    
    try:
        from .demo_interface import main as demo_main
        return demo_main()
    except ImportError:
        QMessageBox.critical(
            None,
            "Ошибка",
            "Демонстрационный файл не найден: demo_interface.py"
        )
        return 1


if __name__ == '__main__':
    # Проверяем аргументы командной строки
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        sys.exit(launch_demo())
    else:
        sys.exit(main())


