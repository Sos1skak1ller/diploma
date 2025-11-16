"""
Мост для интеграции нового интерфейса с существующим кодом.
Перенесён из старого programm/integration_bridge.py и адаптирован под пакет main.
"""

import os
from PyQt5.QtWidgets import QMessageBox


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
            print(f"Запуск существующей детекции для файлов: {files}, уверенность: {confidence}")
            # Пока что используем новую логику
            self.new_interface.controller.start_detection()
            
        except Exception as e:
            QMessageBox.critical(self.new_interface, "Ошибка", f"Ошибка детекции: {str(e)}")
            
    def run_existing_roi_analysis(self, files, roi_type, sensitivity):
        """Запуск существующей логики анализа ROI"""
        
        try:
            print(f"Запуск существующего анализа ROI для файлов: {files}, тип: {roi_type}, чувствительность: {sensitivity}")
            self.new_interface.controller.start_roi_analysis()
            
        except Exception as e:
            QMessageBox.critical(self.new_interface, "Ошибка", f"Ошибка анализа ROI: {str(e)}")
            
    def run_existing_enhancement(self, files, enhance_type, intensity):
        """Запуск существующей логики улучшения качества"""
        
        try:
            print(f"Запуск существующего улучшения для файлов: {files}, тип: {enhance_type}, интенсивность: {intensity}")
            self.new_interface.controller.start_enhancement()
            
        except Exception as e:
            QMessageBox.critical(self.new_interface, "Ошибка", f"Ошибка улучшения: {str(e)}")


def create_integrated_interface():
    """Создает интегрированный интерфейс"""
    
    from .main_interface import MainInterface
    from .new_controller import NewController
    
    # Создаем новый интерфейс
    interface = MainInterface()
    
    # Создаем контроллер и привязываем к интерфейсу
    controller = NewController(interface)
    try:
        interface.controller = controller
    except Exception:
        pass
    
    # Существующая модель (если когда-либо понадобится)
    existing_model = None
    
    # Создаем мост интеграции
    bridge = IntegrationBridge(interface, existing_model)
    
    return interface, controller, bridge


