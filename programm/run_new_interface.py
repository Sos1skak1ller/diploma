#!/usr/bin/env python3
"""
Запускающий файл для нового интерфейса с тремя алгоритмами
"""

import sys
import os
from PyQt5.QtWidgets import QApplication
from main_interface import MainInterface
from new_controller import NewController

def main():
    """Главная функция запуска приложения"""
    
    # Создаем приложение Qt
    app = QApplication(sys.argv)
    app.setApplicationName("Система анализа спутниковых снимков")
    app.setApplicationVersion("2.0")
    
    # Создаем главное окно
    window = MainInterface()
    
    # Создаем контроллер и связываем его с интерфейсом
    controller = NewController(window)
    
    # Показываем окно
    window.show()
    
    # Запускаем главный цикл приложения
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
