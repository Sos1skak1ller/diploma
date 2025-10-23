import os
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, 
                             QVBoxLayout, QHBoxLayout, QSplitter, QLabel, 
                             QPushButton, QSlider, QSpinBox, QLineEdit, 
                             QTextEdit, QFileDialog, QMessageBox, QProgressBar,
                             QFrame, QScrollArea, QGroupBox, QGridLayout,
                             QComboBox, QCheckBox, QListWidget, QListWidgetItem)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRectF, QPointF
from PyQt5.QtGui import QPixmap, QWheelEvent, QMouseEvent, QFont, QPalette, QColor
from view import MyGraphicsView, TableWidget

class MainInterface(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Система анализа спутниковых снимков")
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QTabWidget::pane {
                border: 1px solid #555555;
                background-color: #3c3c3c;
            }
            QTabBar::tab {
                background-color: #555555;
                color: #ffffff;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #0078d4;
            }
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QLineEdit, QSpinBox {
                background-color: #555555;
                color: #ffffff;
                border: 1px solid #777777;
                padding: 4px;
                border-radius: 4px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #777777;
                height: 8px;
                background: #555555;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #0078d4;
                border: 1px solid #0078d4;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #555555;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        self.init_ui()
        # Контроллер будет инициализирован отдельно
        
    def init_ui(self):
        # Создаем центральный виджет с вкладками
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Главный layout
        main_layout = QVBoxLayout(self.central_widget)
        
        # Создаем табы
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Создаем вкладки
        self.create_main_tab()      # Главная - детекция объектов
        self.create_roi_tab()       # Зоны интересов
        self.create_enhance_tab()   # Улучшение качества
        
        # Создаем статус бар
        self.create_status_bar()
        
    def create_status_bar(self):
        self.status_bar = self.statusBar()
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
        
    def create_main_tab(self):
        """Главная вкладка - детекция объектов"""
        main_tab = QWidget()
        self.tab_widget.addTab(main_tab, "Главная - Детекция объектов")
        
        # Основной layout
        main_layout = QHBoxLayout(main_tab)
        
        # Создаем splitter
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Левая панель - управление
        left_panel = self.create_control_panel()
        splitter.addWidget(left_panel)
        
        # Правая панель - изображения и результаты
        right_panel = self.create_image_panel()
        splitter.addWidget(right_panel)
        
        # Настройка пропорций
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        
        # Сохраняем ссылки для доступа из контроллера
        self.main_tab = main_tab
        self.main_splitter = splitter
        
    def create_roi_tab(self):
        """Вкладка зон интересов"""
        roi_tab = QWidget()
        self.tab_widget.addTab(roi_tab, "Зоны интересов")
        
        layout = QHBoxLayout(roi_tab)
        
        # Левая панель управления
        roi_control_panel = self.create_roi_control_panel()
        layout.addWidget(roi_control_panel)
        
        # Правая панель изображений
        roi_image_panel = self.create_roi_image_panel()
        layout.addWidget(roi_image_panel)
        
        self.roi_tab = roi_tab
        
    def create_enhance_tab(self):
        """Вкладка улучшения качества"""
        enhance_tab = QWidget()
        self.tab_widget.addTab(enhance_tab, "Улучшение качества снимков")
        
        layout = QHBoxLayout(enhance_tab)
        
        # Левая панель управления
        enhance_control_panel = self.create_enhance_control_panel()
        layout.addWidget(enhance_control_panel)
        
        # Правая панель изображений
        enhance_image_panel = self.create_enhance_image_panel()
        layout.addWidget(enhance_image_panel)
        
        self.enhance_tab = enhance_tab
        
    def create_control_panel(self):
        """Панель управления для главной вкладки"""
        panel = QFrame()
        panel.setMaximumWidth(350)
        panel.setFrameStyle(QFrame.StyledPanel)
        
        layout = QVBoxLayout(panel)
        
        # Группа координат
        coords_group = QGroupBox("Координаты")
        coords_layout = QGridLayout(coords_group)
        
        coords_layout.addWidget(QLabel("Широта:"), 0, 0)
        self.lat_input = QLineEdit("55.82103")
        coords_layout.addWidget(self.lat_input, 0, 1)
        
        coords_layout.addWidget(QLabel("Долгота:"), 1, 0)
        self.lon_input = QLineEdit("49.16219")
        coords_layout.addWidget(self.lon_input, 1, 1)
        
        coords_layout.addWidget(QLabel("Масштаб:"), 2, 0)
        self.scale_input = QSpinBox()
        self.scale_input.setRange(1, 20)
        self.scale_input.setValue(16)
        coords_layout.addWidget(self.scale_input, 2, 1)
        
        coords_layout.addWidget(QLabel("Уверенность классификатора:"), 3, 0)
        self.confidence_label = QLabel("0.35")
        coords_layout.addWidget(self.confidence_label, 3, 1)
        
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(1, 100)
        self.confidence_slider.setValue(35)
        self.confidence_slider.valueChanged.connect(self.update_confidence_label)
        coords_layout.addWidget(self.confidence_slider, 4, 0, 1, 2)
        
        self.send_coords_btn = QPushButton("Отправить координаты")
        coords_layout.addWidget(self.send_coords_btn, 5, 0, 1, 2)
        
        layout.addWidget(coords_group)
        
        # Группа загрузки файлов
        upload_group = QGroupBox("Загрузка файлов")
        upload_layout = QVBoxLayout(upload_group)
        
        self.upload_area = QLabel("Перетащите файлы сюда")
        self.upload_area.setMinimumHeight(100)
        self.upload_area.setStyleSheet("""
            QLabel {
                border: 2px dashed #777777;
                border-radius: 5px;
                background-color: #404040;
                text-align: center;
                color: #cccccc;
            }
        """)
        self.upload_area.setAlignment(Qt.AlignCenter)
        upload_layout.addWidget(self.upload_area)
        
        self.browse_btn = QPushButton("Выбрать файлы")
        upload_layout.addWidget(self.browse_btn)
        
        self.file_list = QListWidget()
        upload_layout.addWidget(self.file_list)
        
        layout.addWidget(upload_group)
        
        # Группа настроек
        settings_group = QGroupBox("Настройки детекции")
        settings_layout = QVBoxLayout(settings_group)
        
        self.detect_btn = QPushButton("Запустить детекцию")
        settings_layout.addWidget(self.detect_btn)
        
        self.save_results_btn = QPushButton("Сохранить результаты")
        settings_layout.addWidget(self.save_results_btn)
        
        layout.addWidget(settings_group)
        
        layout.addStretch()
        return panel
        
    def create_image_panel(self):
        """Панель изображений для главной вкладки"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel)
        
        layout = QVBoxLayout(panel)
        
        # Изображения
        images_layout = QHBoxLayout()
        
        # Оригинальное изображение
        original_group = QGroupBox("Оригинальное изображение")
        original_layout = QVBoxLayout(original_group)
        self.original_view = MyGraphicsView()
        original_layout.addWidget(self.original_view)
        images_layout.addWidget(original_group)
        
        # Предсказание модели
        prediction_group = QGroupBox("Предсказание модели")
        prediction_layout = QVBoxLayout(prediction_group)
        self.prediction_view = MyGraphicsView()
        prediction_layout.addWidget(self.prediction_view)
        images_layout.addWidget(prediction_group)
        
        layout.addLayout(images_layout)
        
        # Результаты детекции
        results_group = QGroupBox("Найденные объекты")
        results_layout = QVBoxLayout(results_group)
        
        self.results_table = TableWidget()
        self.results_table.setColumnCount(2)
        self.results_table.setHorizontalHeaderLabels(["Объект", "Количество"])
        results_layout.addWidget(self.results_table)
        
        layout.addWidget(results_group)
        
        return panel
        
    def create_roi_control_panel(self):
        """Панель управления для вкладки зон интересов"""
        panel = QFrame()
        panel.setMaximumWidth(350)
        panel.setFrameStyle(QFrame.StyledPanel)
        
        layout = QVBoxLayout(panel)
        
        # Группа настроек ROI
        roi_group = QGroupBox("Настройки зон интересов")
        roi_layout = QVBoxLayout(roi_group)
        
        roi_layout.addWidget(QLabel("Тип анализа:"))
        self.roi_type_combo = QComboBox()
        self.roi_type_combo.addItems(["Автоматическое выделение", "Ручное выделение", "По координатам"])
        roi_layout.addWidget(self.roi_type_combo)
        
        roi_layout.addWidget(QLabel("Чувствительность:"))
        self.roi_sensitivity_slider = QSlider(Qt.Horizontal)
        self.roi_sensitivity_slider.setRange(1, 100)
        self.roi_sensitivity_slider.setValue(50)
        roi_layout.addWidget(self.roi_sensitivity_slider)
        
        self.roi_analyze_btn = QPushButton("Анализировать зоны")
        roi_layout.addWidget(self.roi_analyze_btn)
        
        layout.addWidget(roi_group)
        
        # Группа результатов ROI
        roi_results_group = QGroupBox("Результаты анализа")
        roi_results_layout = QVBoxLayout(roi_results_group)
        
        self.roi_results_list = QListWidget()
        roi_results_layout.addWidget(self.roi_results_list)
        
        layout.addWidget(roi_results_group)
        
        layout.addStretch()
        return panel
        
    def create_roi_image_panel(self):
        """Панель изображений для вкладки зон интересов"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel)
        
        layout = QVBoxLayout(panel)
        
        # Изображения
        images_layout = QHBoxLayout()
        
        # Исходное изображение
        source_group = QGroupBox("Исходное изображение")
        source_layout = QVBoxLayout(source_group)
        self.roi_source_view = MyGraphicsView()
        source_layout.addWidget(self.roi_source_view)
        images_layout.addWidget(source_group)
        
        # ROI анализ
        roi_group = QGroupBox("Анализ зон интересов")
        roi_layout = QVBoxLayout(roi_group)
        self.roi_analysis_view = MyGraphicsView()
        roi_layout.addWidget(self.roi_analysis_view)
        images_layout.addWidget(roi_group)
        
        layout.addLayout(images_layout)
        
        return panel
        
    def create_enhance_control_panel(self):
        """Панель управления для вкладки улучшения качества"""
        panel = QFrame()
        panel.setMaximumWidth(350)
        panel.setFrameStyle(QFrame.StyledPanel)
        
        layout = QVBoxLayout(panel)
        
        # Группа настроек улучшения
        enhance_group = QGroupBox("Настройки улучшения SAR снимков")
        enhance_layout = QVBoxLayout(enhance_group)
        
        enhance_layout.addWidget(QLabel("Тип улучшения:"))
        self.enhance_type_combo = QComboBox()
        self.enhance_type_combo.addItems([
            "Автоматическое улучшение SAR", 
            "SAR подавление шума", 
            "SAR осветление темных областей", 
            "SAR контрастность", 
            "SAR резкость",
            "Комплексное SAR улучшение",
            "AI подавление шума SAR",
            "AI комплексное улучшение SAR"
        ])
        enhance_layout.addWidget(self.enhance_type_combo)
        
        enhance_layout.addWidget(QLabel("Интенсивность:"))
        self.enhance_intensity_slider = QSlider(Qt.Horizontal)
        self.enhance_intensity_slider.setRange(1, 100)
        self.enhance_intensity_slider.setValue(50)
        self.enhance_intensity_slider.valueChanged.connect(self.update_intensity_info)
        enhance_layout.addWidget(self.enhance_intensity_slider)
        
        # Информация о текущих параметрах
        self.intensity_info = QLabel("Интенсивность: 50%")
        self.intensity_info.setStyleSheet("color: #cccccc; font-size: 12px;")
        enhance_layout.addWidget(self.intensity_info)
        
        self.enhance_btn = QPushButton("Улучшить SAR снимок")
        enhance_layout.addWidget(self.enhance_btn)
        
        layout.addWidget(enhance_group)
        
        # Группа информации о параметрах
        info_group = QGroupBox("Информация о параметрах")
        info_layout = QVBoxLayout(info_group)
        
        self.param_info = QTextEdit()
        self.param_info.setMaximumHeight(120)
        self.param_info.setReadOnly(True)
        self.param_info.setStyleSheet("""
            QTextEdit {
                background-color: #404040;
                color: #ffffff;
                border: 1px solid #666666;
                font-size: 11px;
            }
        """)
        self.param_info.setText("Выберите тип улучшения для получения информации о параметрах...")
        info_layout.addWidget(self.param_info)
        
        layout.addWidget(info_group)
        
        # Группа предпросмотра
        preview_group = QGroupBox("Результаты")
        preview_layout = QVBoxLayout(preview_group)
        
        self.enhance_preview_btn = QPushButton("Предпросмотр")
        preview_layout.addWidget(self.enhance_preview_btn)
        
        self.enhance_save_btn = QPushButton("Сохранить улучшенное")
        preview_layout.addWidget(self.enhance_save_btn)
        
        # Информация о качестве
        self.quality_info = QLabel("Метрики качества будут показаны после обработки")
        self.quality_info.setStyleSheet("color: #cccccc; font-size: 11px;")
        self.quality_info.setWordWrap(True)
        preview_layout.addWidget(self.quality_info)
        
        layout.addWidget(preview_group)
        
        # Подключаем обновление информации при изменении типа
        self.enhance_type_combo.currentTextChanged.connect(self.update_enhance_info)
        
        layout.addStretch()
        return panel
        
    def create_enhance_image_panel(self):
        """Панель изображений для вкладки улучшения качества"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.StyledPanel)
        
        layout = QVBoxLayout(panel)
        
        # Изображения
        images_layout = QHBoxLayout()
        
        # Исходное изображение
        original_group = QGroupBox("Исходное изображение")
        original_layout = QVBoxLayout(original_group)
        self.enhance_original_view = MyGraphicsView()
        original_layout.addWidget(self.enhance_original_view)
        images_layout.addWidget(original_group)
        
        # Улучшенное изображение
        enhanced_group = QGroupBox("Улучшенное изображение")
        enhanced_layout = QVBoxLayout(enhanced_group)
        self.enhance_result_view = MyGraphicsView()
        enhanced_layout.addWidget(self.enhance_result_view)
        images_layout.addWidget(enhanced_group)
        
        layout.addLayout(images_layout)
        
        return panel
        
    def update_confidence_label(self, value):
        """Обновляет метку уверенности классификатора"""
        confidence = value / 100.0
        self.confidence_label.setText(f"{confidence:.2f}")
        
    def update_intensity_info(self, value):
        """Обновляет информацию об интенсивности"""
        self.intensity_info.setText(f"Интенсивность: {value}%")
        
    def update_enhance_info(self, enhance_type):
        """Обновляет информацию о параметрах улучшения"""
        info_text = {
            "Автоматическое улучшение SAR": 
                "Автоматически определяет лучший метод улучшения для SAR снимка.\n"
                "Применяет: подавление шума, логарифмическое растяжение, CLAHE.\n"
                "Рекомендуется для большинства радиолокационных снимков.",
                
            "SAR подавление шума": 
                "Специализированное подавление шума для SAR изображений.\n"
                "Использует: Non-local Means Denoising + Bilateral Filter.\n"
                "Эффективно для удаления зернистости и артефактов.",
                
            "SAR осветление темных областей": 
                "Осветление темных областей в SAR снимках.\n"
                "Применяет: гамма-коррекцию, логарифмическое растяжение.\n"
                "Улучшает видимость деталей в теневых областях.",
                
            "SAR контрастность": 
                "Улучшение контраста для радиолокационных снимков.\n"
                "Использует: CLAHE (Contrast Limited Adaptive Histogram Equalization).\n"
                "Подчеркивает различия в интенсивности отражений.",
                
            "SAR резкость": 
                "Увеличение резкости SAR изображений.\n"
                "Применяет: специальные ядра свертки для SAR данных.\n"
                "Улучшает четкость границ объектов.",
                
            "Комплексное SAR улучшение": 
                "Полный пайплайн улучшения SAR снимков.\n"
                "Включает: подавление шума + осветление + контраст + резкость.\n"
                "Максимальное качество для критически важных снимков.",
                
            "AI подавление шума SAR": 
                "Нейронная сеть для подавления шума SAR изображений.\n"
                "Использует: DnCNN/RIDNet архитектуры с ONNX Runtime.\n"
                "Автоматический fallback на традиционные методы.",
                
            "AI комплексное улучшение SAR": 
                "Полный AI-пайплайн улучшения SAR снимков.\n"
                "Включает: AI-подавление шума + адаптивное осветление + CLAHE.\n"
                "Современные нейронные сети для максимального качества."
        }
        
        self.param_info.setText(info_text.get(enhance_type, "Информация недоступна"))
        
    def update_progress(self, progress, filename=""):
        """Обновляет прогресс бар"""
        self.progress_bar.setValue(progress)
        if filename:
            self.status_bar.showMessage(f"Обработка: {filename}")
        else:
            self.status_bar.showMessage("Обработка завершена")
            
    def show_error(self, message):
        """Показывает сообщение об ошибке"""
        QMessageBox.warning(self, "Ошибка", message)
        
    def closeEvent(self, event):
        """Обработка закрытия приложения"""
        # Очистка временных файлов
        if os.path.exists('tmp'):
            for file in os.listdir('tmp'):
                try:
                    os.remove(os.path.join('tmp', file))
                except:
                    pass
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = MainInterface()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
