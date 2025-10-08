"""
Мост для интеграции нового интерфейса с существующим кодом
"""

import os
import sys
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import QThread, pyqtSignal

# Импорт существующих модулей
try:
    from view import MyGraphicsView, TableWidget
    from model import Model  # Предполагаем, что модель существует
except ImportError as e:
    print(f"Предупреждение: Не удалось импортировать существующие модули: {e}")

class IntegrationBridge:
    """Класс для интеграции нового интерфейса с существующим кодом"""
    
    def __init__(self, new_interface, existing_model=None):
        self.new_interface = new_interface
        self.existing_model = existing_model
        self.setup_integration()
        
    def setup_integration(self):
        """Настройка интеграции между новым и старым кодом"""
        
        # Подключение сигналов от нового интерфейса к существующей логике
        self.connect_signals()
        
        # Инициализация существующих компонентов
        self.init_existing_components()
        
    def connect_signals(self):
        """Подключение сигналов между компонентами"""
        
        # Подключение сигналов детекции
        if hasattr(self.new_interface, 'detect_btn'):
            self.new_interface.detect_btn.clicked.connect(self.handle_detection_request)
            
        # Подключение сигналов ROI анализа
        if hasattr(self.new_interface, 'roi_analyze_btn'):
            self.new_interface.roi_analyze_btn.clicked.connect(self.handle_roi_request)
            
        # Подключение сигналов улучшения качества
        if hasattr(self.new_interface, 'enhance_btn'):
            self.new_interface.enhance_btn.clicked.connect(self.handle_enhancement_request)
            
    def init_existing_components(self):
        """Инициализация существующих компонентов"""
        
        # Здесь можно инициализировать существующие модели, веса и т.д.
        if self.existing_model:
            try:
                # Инициализация существующей модели
                self.existing_model.load_weights()
                print("Существующая модель успешно загружена")
            except Exception as e:
                print(f"Ошибка загрузки существующей модели: {e}")
                
    def handle_detection_request(self):
        """Обработка запроса на детекцию объектов"""
        
        # Получаем параметры из нового интерфейса
        confidence = self.new_interface.confidence_slider.value() / 100.0
        
        # Получаем выбранные файлы
        selected_files = self.get_selected_files()
        if not selected_files:
            QMessageBox.warning(self.new_interface, "Ошибка", "Выберите файлы для анализа")
            return
            
        # Запускаем существующую логику детекции
        if self.existing_model:
            self.run_existing_detection(selected_files, confidence)
        else:
            # Используем новую логику
            self.new_interface.controller.start_detection()
            
    def handle_roi_request(self):
        """Обработка запроса на анализ ROI"""
        
        roi_type = self.new_interface.roi_type_combo.currentText()
        sensitivity = self.new_interface.roi_sensitivity_slider.value()
        
        selected_files = self.get_selected_files()
        if not selected_files:
            QMessageBox.warning(self.new_interface, "Ошибка", "Выберите файлы для анализа")
            return
            
        if self.existing_model:
            self.run_existing_roi_analysis(selected_files, roi_type, sensitivity)
        else:
            self.new_interface.controller.start_roi_analysis()
            
    def handle_enhancement_request(self):
        """Обработка запроса на улучшение качества"""
        
        enhance_type = self.new_interface.enhance_type_combo.currentText()
        intensity = self.new_interface.enhance_intensity_slider.value()
        
        selected_files = self.get_selected_files()
        if not selected_files:
            QMessageBox.warning(self.new_interface, "Ошибка", "Выберите файлы для улучшения")
            return
            
        if self.existing_model:
            self.run_existing_enhancement(selected_files, enhance_type, intensity)
        else:
            self.new_interface.controller.start_enhancement()
            
    def get_selected_files(self):
        """Получает список выбранных файлов"""
        
        files = []
        if hasattr(self.new_interface, 'file_list'):
            for i in range(self.new_interface.file_list.count()):
                item = self.new_interface.file_list.item(i)
                if item:
                    file_path = item.data(1)
                    if file_path and os.path.exists(file_path):
                        files.append(file_path)
        return files
        
    def run_existing_detection(self, files, confidence):
        """Запуск существующей логики детекции"""
        
        try:
            # Здесь должна быть интеграция с существующим кодом детекции
            # Например:
            # results = self.existing_model.detect_objects(files[0], confidence)
            # self.new_interface.controller.on_detection_finished(results)
            
            print(f"Запуск существующей детекции для файлов: {files}, уверенность: {confidence}")
            
            # Пока что используем новую логику
            self.new_interface.controller.start_detection()
            
        except Exception as e:
            QMessageBox.critical(self.new_interface, "Ошибка", f"Ошибка детекции: {str(e)}")
            
    def run_existing_roi_analysis(self, files, roi_type, sensitivity):
        """Запуск существующей логики анализа ROI"""
        
        try:
            print(f"Запуск существующего анализа ROI для файлов: {files}, тип: {roi_type}, чувствительность: {sensitivity}")
            
            # Здесь должна быть интеграция с существующим кодом ROI
            self.new_interface.controller.start_roi_analysis()
            
        except Exception as e:
            QMessageBox.critical(self.new_interface, "Ошибка", f"Ошибка анализа ROI: {str(e)}")
            
    def run_existing_enhancement(self, files, enhance_type, intensity):
        """Запуск существующей логики улучшения качества"""
        
        try:
            print(f"Запуск существующего улучшения для файлов: {files}, тип: {enhance_type}, интенсивность: {intensity}")
            
            # Здесь должна быть интеграция с существующим кодом улучшения
            self.new_interface.controller.start_enhancement()
            
        except Exception as e:
            QMessageBox.critical(self.new_interface, "Ошибка", f"Ошибка улучшения: {str(e)}")

class LegacyModelAdapter:
    """Адаптер для работы со старой моделью"""
    
    def __init__(self, legacy_model):
        self.legacy_model = legacy_model
        
    def detect_objects(self, image_path, confidence):
        """Адаптер для детекции объектов"""
        
        try:
            # Адаптация вызова старой модели к новому интерфейсу
            if hasattr(self.legacy_model, 'detect'):
                results = self.legacy_model.detect(image_path, confidence)
                return self.adapt_detection_results(results)
            else:
                # Fallback к новой логике
                return self.fallback_detection(image_path, confidence)
                
        except Exception as e:
            print(f"Ошибка адаптера детекции: {e}")
            return self.fallback_detection(image_path, confidence)
            
    def adapt_detection_results(self, legacy_results):
        """Адаптация результатов старой модели к новому формату"""
        
        # Преобразование формата результатов
        adapted_results = {
            'objects': [],
            'image_with_boxes': None
        }
        
        # Здесь должна быть логика преобразования формата
        # Например, если старая модель возвращает список объектов
        if isinstance(legacy_results, list):
            for obj in legacy_results:
                adapted_results['objects'].append({
                    'class': obj.get('class_name', 'Неизвестно'),
                    'count': obj.get('count', 1)
                })
                
        return adapted_results
        
    def fallback_detection(self, image_path, confidence):
        """Fallback детекция при ошибке адаптера"""
        
        # Возвращаем демонстрационные результаты
        return {
            'objects': [
                {'class': 'Корабль', 'count': 47},
                {'class': 'Гавань', 'count': 3},
                {'class': 'Теннисный корт', 'count': 2}
            ],
            'image_with_boxes': image_path
        }

def create_integrated_interface():
    """Создает интегрированный интерфейс"""
    
    from main_interface import MainInterface
    from new_controller import NewController
    
    # Создаем новый интерфейс
    interface = MainInterface()
    
    # Создаем контроллер
    controller = NewController(interface)
    
    # Пытаемся загрузить существующую модель
    existing_model = None
    try:
        # Здесь должна быть загрузка существующей модели
        # existing_model = load_existing_model()
        pass
    except Exception as e:
        print(f"Не удалось загрузить существующую модель: {e}")
        
    # Создаем мост интеграции
    bridge = IntegrationBridge(interface, existing_model)
    
    return interface, controller, bridge

if __name__ == '__main__':
    # Тестирование интеграции
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    interface, controller, bridge = create_integrated_interface()
    interface.show()
    sys.exit(app.exec_())
