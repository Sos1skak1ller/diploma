import os
import sys
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QPixmap
import model

class DetectionWorker(QThread):
    """Поток для выполнения детекции объектов"""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, image_path, confidence):
        super().__init__()
        self.image_path = image_path
        self.confidence = confidence
        
    def run(self):
        try:
            # Здесь должна быть логика детекции из модели
            # Пока что имитируем процесс
            self.progress.emit(25, "Загрузка изображения...")
            # model.load_image(self.image_path)
            
            self.progress.emit(50, "Выполнение детекции...")
            # results = model.detect_objects(self.image_path, self.confidence)
            
            # Имитируем результаты
            results = {
                'objects': [
                    {'class': 'Корабль', 'count': 47},
                    {'class': 'Гавань', 'count': 3},
                    {'class': 'Теннисный корт', 'count': 2}
                ],
                'image_with_boxes': self.image_path  # Путь к изображению с боксами
            }
            
            self.progress.emit(100, "Детекция завершена")
            self.finished.emit(results)
            
        except Exception as e:
            self.error.emit(str(e))

class ROIWorker(QThread):
    """Поток для анализа зон интересов"""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, image_path, roi_type, sensitivity):
        super().__init__()
        self.image_path = image_path
        self.roi_type = roi_type
        self.sensitivity = sensitivity
        
    def run(self):
        try:
            self.progress.emit(25, "Анализ зон интересов...")
            # Здесь должна быть логика анализа ROI
            
            self.progress.emit(75, "Обработка результатов...")
            
            # Имитируем результаты
            results = {
                'roi_regions': [
                    {'type': 'Портовая зона', 'confidence': 0.95},
                    {'type': 'Жилая зона', 'confidence': 0.87},
                    {'type': 'Промышленная зона', 'confidence': 0.72}
                ],
                'roi_image': self.image_path
            }
            
            self.progress.emit(100, "Анализ завершен")
            self.finished.emit(results)
            
        except Exception as e:
            self.error.emit(str(e))

class EnhancementWorker(QThread):
    """Поток для улучшения качества изображений"""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, image_path, enhance_type, intensity):
        super().__init__()
        self.image_path = image_path
        self.enhance_type = enhance_type
        self.intensity = intensity
        
    def run(self):
        try:
            self.progress.emit(25, "Анализ качества изображения...")
            # Здесь должна быть логика улучшения
            
            self.progress.emit(75, "Применение улучшений...")
            
            # Имитируем результаты
            results = {
                'enhanced_image': self.image_path,  # Путь к улучшенному изображению
                'quality_metrics': {
                    'contrast': 0.85,
                    'brightness': 0.78,
                    'sharpness': 0.92
                }
            }
            
            self.progress.emit(100, "Улучшение завершено")
            self.finished.emit(results)
            
        except Exception as e:
            self.error.emit(str(e))

class NewController:
    def __init__(self, view):
        self.view = view
        self.current_workers = []
        self.setup_connections()
        
    def setup_connections(self):
        """Настройка соединений между интерфейсом и контроллером"""
        
        # Главная вкладка - детекция объектов
        self.view.browse_btn.clicked.connect(self.browse_files_main)
        self.view.detect_btn.clicked.connect(self.start_detection)
        self.view.send_coords_btn.clicked.connect(self.send_coordinates)
        self.view.save_results_btn.clicked.connect(self.save_detection_results)
        
        # Вкладка зон интересов
        self.view.roi_analyze_btn.clicked.connect(self.start_roi_analysis)
        
        # Вкладка улучшения качества
        self.view.enhance_btn.clicked.connect(self.start_enhancement)
        self.view.enhance_preview_btn.clicked.connect(self.preview_enhancement)
        self.view.enhance_save_btn.clicked.connect(self.save_enhanced_image)
        
        # Настройка drag and drop
        self.setup_drag_drop()
        
    def setup_drag_drop(self):
        """Настройка drag and drop функциональности"""
        # Здесь должна быть реализация drag and drop
        # Пока что используем только кнопку выбора файлов
        pass
        
    def browse_files_main(self):
        """Выбор файлов для главной вкладки"""
        files, _ = QFileDialog.getOpenFileNames(
            self.view,
            "Выберите изображения",
            "",
            "Изображения (*.jpg *.jpeg *.png *.bmp *.tiff);;Все файлы (*)"
        )
        
        if files:
            self.view.file_list.clear()
            for file in files:
                item = QListWidgetItem(os.path.basename(file))
                item.setData(1, file)  # Сохраняем полный путь
                self.view.file_list.addItem(item)
                
    def send_coordinates(self):
        """Отправка координат (имитация)"""
        lat = self.view.lat_input.text()
        lon = self.view.lon_input.text()
        scale = self.view.scale_input.value()
        
        QMessageBox.information(
            self.view,
            "Координаты отправлены",
            f"Широта: {lat}, Долгота: {lon}, Масштаб: {scale}"
        )
        
    def start_detection(self):
        """Запуск детекции объектов"""
        if self.view.file_list.count() == 0:
            self.view.show_error("Выберите изображения для анализа")
            return
            
        # Получаем первый выбранный файл
        item = self.view.file_list.item(0)
        if not item:
            return
            
        image_path = item.data(1)
        confidence = self.view.confidence_slider.value() / 100.0
        
        # Отключаем кнопки во время обработки
        self.set_ui_enabled(False)
        
        # Запускаем поток детекции
        worker = DetectionWorker(image_path, confidence)
        worker.progress.connect(self.view.update_progress)
        worker.finished.connect(self.on_detection_finished)
        worker.error.connect(self.on_detection_error)
        
        self.current_workers.append(worker)
        worker.start()
        
        # Показываем оригинальное изображение
        self.view.original_view.set_image(image_path)
        
    def on_detection_finished(self, results):
        """Обработка завершения детекции"""
        self.set_ui_enabled(True)
        self.view.progress_bar.setVisible(False)
        
        # Обновляем таблицу результатов
        self.view.results_table.setRowCount(len(results['objects']))
        for i, obj in enumerate(results['objects']):
            self.view.results_table.setItem(i, 0, QTableWidgetItem(obj['class']))
            self.view.results_table.setItem(i, 1, QTableWidgetItem(str(obj['count'])))
            
        # Показываем изображение с детекцией
        if 'image_with_boxes' in results:
            self.view.prediction_view.set_image(results['image_with_boxes'])
            
    def on_detection_error(self, error_message):
        """Обработка ошибки детекции"""
        self.set_ui_enabled(True)
        self.view.progress_bar.setVisible(False)
        self.view.show_error(f"Ошибка детекции: {error_message}")
        
    def start_roi_analysis(self):
        """Запуск анализа зон интересов"""
        if self.view.file_list.count() == 0:
            self.view.show_error("Выберите изображения для анализа")
            return
            
        item = self.view.file_list.item(0)
        if not item:
            return
            
        image_path = item.data(1)
        roi_type = self.view.roi_type_combo.currentText()
        sensitivity = self.view.roi_sensitivity_slider.value()
        
        self.set_ui_enabled(False)
        
        worker = ROIWorker(image_path, roi_type, sensitivity)
        worker.progress.connect(self.view.update_progress)
        worker.finished.connect(self.on_roi_finished)
        worker.error.connect(self.on_roi_error)
        
        self.current_workers.append(worker)
        worker.start()
        
        # Показываем исходное изображение
        self.view.roi_source_view.set_image(image_path)
        
    def on_roi_finished(self, results):
        """Обработка завершения анализа ROI"""
        self.set_ui_enabled(True)
        self.view.progress_bar.setVisible(False)
        
        # Обновляем список результатов ROI
        self.view.roi_results_list.clear()
        for roi in results['roi_regions']:
            item_text = f"{roi['type']} (уверенность: {roi['confidence']:.2f})"
            self.view.roi_results_list.addItem(item_text)
            
        # Показываем результат анализа
        if 'roi_image' in results:
            self.view.roi_analysis_view.set_image(results['roi_image'])
            
    def on_roi_error(self, error_message):
        """Обработка ошибки анализа ROI"""
        self.set_ui_enabled(True)
        self.view.progress_bar.setVisible(False)
        self.view.show_error(f"Ошибка анализа ROI: {error_message}")
        
    def start_enhancement(self):
        """Запуск улучшения качества"""
        if self.view.file_list.count() == 0:
            self.view.show_error("Выберите изображения для улучшения")
            return
            
        item = self.view.file_list.item(0)
        if not item:
            return
            
        image_path = item.data(1)
        enhance_type = self.view.enhance_type_combo.currentText()
        intensity = self.view.enhance_intensity_slider.value()
        
        self.set_ui_enabled(False)
        
        worker = EnhancementWorker(image_path, enhance_type, intensity)
        worker.progress.connect(self.view.update_progress)
        worker.finished.connect(self.on_enhancement_finished)
        worker.error.connect(self.on_enhancement_error)
        
        self.current_workers.append(worker)
        worker.start()
        
        # Показываем исходное изображение
        self.view.enhance_original_view.set_image(image_path)
        
    def on_enhancement_finished(self, results):
        """Обработка завершения улучшения"""
        self.set_ui_enabled(True)
        self.view.progress_bar.setVisible(False)
        
        # Показываем улучшенное изображение
        if 'enhanced_image' in results:
            self.view.enhance_result_view.set_image(results['enhanced_image'])
            
        # Показываем метрики качества
        if 'quality_metrics' in results:
            metrics = results['quality_metrics']
            message = f"Метрики качества:\n"
            message += f"Контрастность: {metrics['contrast']:.2f}\n"
            message += f"Яркость: {metrics['brightness']:.2f}\n"
            message += f"Резкость: {metrics['sharpness']:.2f}"
            QMessageBox.information(self.view, "Результаты улучшения", message)
            
    def on_enhancement_error(self, error_message):
        """Обработка ошибки улучшения"""
        self.set_ui_enabled(True)
        self.view.progress_bar.setVisible(False)
        self.view.show_error(f"Ошибка улучшения: {error_message}")
        
    def preview_enhancement(self):
        """Предпросмотр улучшений"""
        QMessageBox.information(self.view, "Предпросмотр", "Функция предпросмотра в разработке")
        
    def save_enhanced_image(self):
        """Сохранение улучшенного изображения"""
        file_path, _ = QFileDialog.getSaveFileName(
            self.view,
            "Сохранить улучшенное изображение",
            "",
            "Изображения (*.jpg *.png);;Все файлы (*)"
        )
        
        if file_path:
            QMessageBox.information(self.view, "Сохранение", f"Изображение сохранено: {file_path}")
            
    def save_detection_results(self):
        """Сохранение результатов детекции"""
        file_path, _ = QFileDialog.getSaveFileName(
            self.view,
            "Сохранить результаты детекции",
            "",
            "Текстовые файлы (*.txt);;Все файлы (*)"
        )
        
        if file_path:
            QMessageBox.information(self.view, "Сохранение", f"Результаты сохранены: {file_path}")
            
    def set_ui_enabled(self, enabled):
        """Включение/отключение элементов интерфейса"""
        # Главная вкладка
        self.view.detect_btn.setEnabled(enabled)
        self.view.browse_btn.setEnabled(enabled)
        self.view.send_coords_btn.setEnabled(enabled)
        self.view.save_results_btn.setEnabled(enabled)
        self.view.confidence_slider.setEnabled(enabled)
        
        # Вкладка ROI
        self.view.roi_analyze_btn.setEnabled(enabled)
        self.view.roi_type_combo.setEnabled(enabled)
        self.view.roi_sensitivity_slider.setEnabled(enabled)
        
        # Вкладка улучшения
        self.view.enhance_btn.setEnabled(enabled)
        self.view.enhance_type_combo.setEnabled(enabled)
        self.view.enhance_intensity_slider.setEnabled(enabled)
        self.view.enhance_preview_btn.setEnabled(enabled)
        self.view.enhance_save_btn.setEnabled(enabled)
        
        if not enabled:
            self.view.progress_bar.setVisible(True)
        else:
            self.view.progress_bar.setVisible(False)
