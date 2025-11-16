import sys
import os
from PyQt5.QtWidgets import QApplication, QMessageBox, QListWidgetItem
from .main_interface import MainInterface
from .new_controller import NewController


class DemoController(NewController):
    """Демонстрационный контроллер с имитацией работы алгоритмов"""
    
    def __init__(self, view):
        super().__init__(view)
        self.demo_mode = True
        
    def start_detection(self):
        """Демонстрационная детекция объектов"""
        if self.view.file_list.count() == 0:
            # Добавляем демонстрационный файл
            demo_file = self._resolve_demo_image("1.jpg")
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
            demo_file = self._resolve_demo_image("1.jpg")
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
            demo_file = self._resolve_demo_image("9.jpg")
            if os.path.exists(demo_file):
                item = QListWidgetItem(os.path.basename(demo_file))
                item.setData(1, demo_file)
                self.view.file_list.addItem(item)
            else:
                self.view.show_error("Выберите изображения для улучшения")
                return
                
        super().start_enhancement()

    def _resolve_demo_image(self, name: str) -> str:
        """
        Ищет демо-изображение относительно пакета main/interface.
        Ожидаем структуру main/interface/images/<name>.
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        img_dir = os.path.join(base_dir, "images")
        return os.path.join(img_dir, name)


def setup_demo_data(interface):
    """Настройка демонстрационных данных"""
    
    demo_images = [
        "1.jpg",
        "9.jpg",
    ]

    # Разрешаем пути относительно пакета
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(base_dir, "images")
    
    for name in demo_images:
        img_path = os.path.join(img_dir, name)
        if os.path.exists(img_path):
            item = QListWidgetItem(os.path.basename(img_path))
            item.setData(1, img_path)
            interface.file_list.addItem(item)
            
    # Настраиваем демонстрационные значения
    interface.lat_input.setText("55.82103")
    interface.lon_input.setText("49.16219")
    interface.scale_input.setValue(16)
    interface.confidence_slider.setValue(35)
    
    # Загружаем первое изображение в различные вкладки
    first_img = os.path.join(img_dir, "1.jpg")
    second_img = os.path.join(img_dir, "9.jpg")
    if os.path.exists(first_img):
        interface.original_view.set_image(first_img)
        interface.roi_source_view.set_image(first_img)
        if os.path.exists(second_img):
            interface.enhance_original_view.set_image(second_img)
        else:
            interface.enhance_original_view.set_image(first_img)


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


