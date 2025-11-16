import sys
import os
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import QTimer
from main_interface import MainInterface
from new_controller import NewController

class DemoController(NewController):
    """Демонстрационный контроллер с имитацией работы алгоритмов"""
    
    def __init__(self, view):
        super().__init__(view)
        self.demo_mode = True
        
    def start_detection(self):
        """Демонстрационная детекция объектов"""
        if self.view.file_list.count() == 0:
            # Добавляем демонстрационный файл
            demo_file = "/Users/sosiska_killer/Documents/diplom/programm/images/1.jpg"
            if os.path.exists(demo_file):
                item = QListWidgetItem(os.path.basename(demo_file))
                item.setData(1, demo_file)
                self.view.file_list.addItem(item)
            else:
                self.view.show_error("Выберите изображения для анализа")
                return
        
        super().start_detection()
        
    def start_roi_analysis(self):
        """Демонстрационный анализ зон интересов"""
        if self.view.file_list.count() == 0:
            # Добавляем демонстрационный файл
            demo_file = "/Users/sosiska_killer/Documents/diplom/programm/images/2.jpg"
            if os.path.exists(demo_file):
                item = QListWidgetItem(os.path.basename(demo_file))
                item.setData(1, demo_file)
                self.view.file_list.addItem(item)
            else:
                self.view.show_error("Выберите изображения для анализа")
                return
                
        super().start_roi_analysis()
        
    def start_enhancement(self):
        """Демонстрационное улучшение качества"""
        if self.view.file_list.count() == 0:
            # Добавляем демонстрационный файл
            demo_file = "/Users/sosiska_killer/Documents/diplom/programm/images/3.jpg"
            if os.path.exists(demo_file):
                item = QListWidgetItem(os.path.basename(demo_file))
                item.setData(1, demo_file)
                self.view.file_list.addItem(item)
            else:
                self.view.show_error("Выберите изображения для улучшения")
                return
                
        super().start_enhancement()

def setup_demo_data(interface):
    """Настройка демонстрационных данных"""
    
    # Добавляем демонстрационные изображения в список файлов
    demo_images = [
        "/Users/sosiska_killer/Documents/diplom/programm/images/1.jpg",
        "/Users/sosiska_killer/Documents/diplom/programm/images/2.jpg",
        "/Users/sosiska_killer/Documents/diplom/programm/images/3.jpg"
    ]
    
    for img_path in demo_images:
        if os.path.exists(img_path):
            from PyQt5.QtWidgets import QListWidgetItem
            item = QListWidgetItem(os.path.basename(img_path))
            item.setData(1, img_path)
            interface.file_list.addItem(item)
            
    # Настраиваем демонстрационные значения
    interface.lat_input.setText("55.82103")
    interface.lon_input.setText("49.16219")
    interface.scale_input.setValue(16)
    interface.confidence_slider.setValue(35)
    
    # Загружаем первое изображение в оригинальный вид
    if os.path.exists(demo_images[0]):
        interface.original_view.set_image(demo_images[0])
        interface.roi_source_view.set_image(demo_images[1] if len(demo_images) > 1 else demo_images[0])
        interface.enhance_original_view.set_image(demo_images[2] if len(demo_images) > 2 else demo_images[0])

def main():
    """Главная функция демонстрации"""
    
    print("Запуск демонстрации интерфейса...")
    
    # Создаем приложение Qt
    app = QApplication(sys.argv)
    app.setApplicationName("Демонстрация системы анализа спутниковых снимков")
    
    # Создаем главное окно
    window = MainInterface()
    
    # Создаем демонстрационный контроллер
    controller = DemoController(window)
    
    # Настраиваем демонстрационные данные
    setup_demo_data(window)
    
    # Показываем информационное сообщение
    QMessageBox.information(
        window,
        "Демонстрационный режим",
        "Запущена демонстрация интерфейса с тремя алгоритмами:\n\n"
        "1. Главная - Детекция объектов\n"
        "2. Зоны интересов - Анализ областей\n"
        "3. Улучшение качества - Обработка изображений\n\n"
        "Демонстрационные изображения уже загружены.\n"
        "Попробуйте запустить различные алгоритмы!"
    )
    
    # Показываем окно
    window.show()
    
    # Запускаем главный цикл приложения
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
