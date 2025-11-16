"""
Улучшенные компоненты для интерфейса
"""

import os
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QScrollArea, QFrame, QSizePolicy,
                             QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
                             QMessageBox, QFileDialog, QListWidget, QListWidgetItem,
                             QSlider, QSpinBox, QComboBox, QCheckBox, QGroupBox,
                             QGridLayout, QTextEdit, QProgressBar)
from PyQt5.QtCore import Qt, pyqtSignal, QRectF, QPointF, QTimer
from PyQt5.QtGui import QPixmap, QWheelEvent, QMouseEvent, QFont, QPainter, QPen

class DragDropWidget(QFrame):
    """Виджет для drag and drop файлов"""
    files_dropped = pyqtSignal(list)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(120)
        self.setFrameStyle(QFrame.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                border: 2px dashed #777777;
                border-radius: 10px;
                background-color: #404040;
                color: #cccccc;
            }
            QFrame:hover {
                border-color: #0078d4;
                background-color: #4a4a4a;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        
        # Иконка облака
        self.icon_label = QLabel("☁️")
        self.icon_label.setAlignment(Qt.AlignCenter)
        self.icon_label.setStyleSheet("font-size: 24px;")
        layout.addWidget(self.icon_label)
        
        # Текст
        self.text_label = QLabel("Перетащите файлы сюда")
        self.text_label.setAlignment(Qt.AlignCenter)
        self.text_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(self.text_label)
        
        # Подзаголовок
        self.subtitle_label = QLabel("Лимит 200MB на файл • PNG, JPG, JPEG")
        self.subtitle_label.setAlignment(Qt.AlignCenter)
        self.subtitle_label.setStyleSheet("font-size: 12px; color: #aaaaaa;")
        layout.addWidget(self.subtitle_label)
        
        # Кнопка выбора файлов
        self.browse_btn = QPushButton("Выбрать файлы")
        self.browse_btn.setMaximumWidth(150)
        layout.addWidget(self.browse_btn, alignment=Qt.AlignCenter)
        
        self.browse_btn.clicked.connect(self.browse_files)
        
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
            self.setStyleSheet("""
                QFrame {
                    border: 2px dashed #0078d4;
                    border-radius: 10px;
                    background-color: #4a4a4a;
                    color: #ffffff;
                }
            """)
        else:
            event.ignore()
            
    def dragLeaveEvent(self, event):
        self.setStyleSheet("""
            QFrame {
                border: 2px dashed #777777;
                border-radius: 10px;
                background-color: #404040;
                color: #cccccc;
            }
        """)
        
    def dropEvent(self, event):
        files = []
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if os.path.isfile(file_path):
                files.append(file_path)
                
        if files:
            self.files_dropped.emit(files)
            
        self.setStyleSheet("""
            QFrame {
                border: 2px dashed #777777;
                border-radius: 10px;
                background-color: #404040;
                color: #cccccc;
            }
        """)
        
    def browse_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Выберите изображения",
            "",
            "Изображения (*.jpg *.jpeg *.png *.bmp *.tiff);;Все файлы (*)"
        )
        if files:
            self.files_dropped.emit(files)

class EnhancedGraphicsView(QGraphicsView):
    """Улучшенный виджет для отображения изображений с дополнительными функциями"""
    
    image_loaded = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        
        # Настройки взаимодействия
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Настройки масштабирования
        self.zoom_factor = 1.25
        self.min_zoom = 0.1
        self.max_zoom = 20.0
        self.current_scale = 1.0
        
        # Переменные для панорамирования
        self._pan_start = QPointF()
        self._panning = False
        
        # Настройка стиля
        self.setStyleSheet("""
            QGraphicsView {
                border: 1px solid #555555;
                background-color: #2b2b2b;
            }
        """)
        
    def set_image(self, image_path):
        """Загружает и отображает изображение"""
        if not os.path.exists(image_path):
            print(f"Файл не найден: {image_path}")
            return
            
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            print(f"Ошибка загрузки изображения: {image_path}")
            return
            
        self.pixmap_item.setPixmap(pixmap)
        self.scene.setSceneRect(QRectF(pixmap.rect()))
        
        # Подгоняем изображение по размеру
        self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
        self.current_scale = 1.0
        
        self.image_loaded.emit(image_path)
        
    def wheelEvent(self, event: QWheelEvent):
        """Обработка масштабирования колесиком мыши"""
        # Удерживаем Ctrl для масштабирования
        if event.modifiers() == Qt.ControlModifier:
            zoom_in = event.angleDelta().y() > 0
            
            if zoom_in and self.current_scale < self.max_zoom:
                self.scale(self.zoom_factor, self.zoom_factor)
                self.current_scale *= self.zoom_factor
            elif not zoom_in and self.current_scale > self.min_zoom:
                self.scale(1 / self.zoom_factor, 1 / self.zoom_factor)
                self.current_scale /= self.zoom_factor
        else:
            super().wheelEvent(event)
            
    def mousePressEvent(self, event: QMouseEvent):
        """Обработка нажатий мыши"""
        if event.button() == Qt.MiddleButton or (event.button() == Qt.LeftButton and event.modifiers() == Qt.ShiftModifier):
            self._pan_start = event.pos()
            self._panning = True
            self.setCursor(Qt.ClosedHandCursor)
        else:
            super().mousePressEvent(event)
            
    def mouseMoveEvent(self, event: QMouseEvent):
        """Обработка движения мыши"""
        if self._panning:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
        else:
            super().mouseMoveEvent(event)
            
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Обработка отпускания мыши"""
        if event.button() == Qt.MiddleButton or event.button() == Qt.LeftButton:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
        else:
            super().mouseReleaseEvent(event)
            
    def fit_to_view(self):
        """Подгоняет изображение под размер окна"""
        if self.pixmap_item.pixmap().isNull():
            return
        self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
        self.current_scale = 1.0
        
    def reset_zoom(self):
        """Сбрасывает масштаб"""
        self.setTransform(self.transform().scale(1/self.current_scale, 1/self.current_scale))
        self.current_scale = 1.0
        self.fit_to_view()

class FileListWidget(QListWidget):
    """Улучшенный виджет для отображения списка файлов"""
    
    file_selected = pyqtSignal(str)
    file_removed = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QListWidget {
                border: 1px solid #555555;
                background-color: #3c3c3c;
                color: #ffffff;
                selection-background-color: #0078d4;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #555555;
            }
            QListWidget::item:hover {
                background-color: #4a4a4a;
            }
        """)
        
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
        
    def add_file(self, file_path):
        """Добавляет файл в список"""
        if os.path.exists(file_path):
            file_name = os.path.basename(file_path)
            file_size = self.get_file_size(file_path)
            
            item = QListWidgetItem(f"{file_name} ({file_size})")
            item.setData(1, file_path)  # Полный путь
            item.setData(2, file_name)  # Имя файла
            item.setData(3, file_size)  # Размер
            
            self.addItem(item)
            
    def remove_selected(self):
        """Удаляет выбранный файл"""
        current_item = self.currentItem()
        if current_item:
            file_path = current_item.data(1)
            self.takeItem(self.row(current_item))
            self.file_removed.emit(file_path)
            
    def get_file_size(self, file_path):
        """Возвращает размер файла в удобном формате"""
        size = os.path.getsize(file_path)
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f}{unit}"
            size /= 1024.0
        return f"{size:.1f}TB"
        
    def show_context_menu(self, position):
        """Показывает контекстное меню"""
        item = self.itemAt(position)
        if item:
            # Здесь можно добавить контекстное меню
            pass

class ProgressWidget(QWidget):
    """Виджет для отображения прогресса"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("")
        self.status_label.setVisible(False)
        layout.addWidget(self.status_label)
        
    def show_progress(self, value, text=""):
        """Показывает прогресс"""
        self.progress_bar.setValue(value)
        self.status_label.setText(text)
        self.progress_bar.setVisible(True)
        self.status_label.setVisible(True)
        
    def hide_progress(self):
        """Скрывает прогресс"""
        self.progress_bar.setVisible(False)
        self.status_label.setVisible(False)

class ResultsTable(QWidget):
    """Виджет для отображения результатов"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Заголовок
        header_label = QLabel("Результаты анализа")
        header_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(header_label)
        
        # Таблица результатов
        from view import TableWidget
        self.table = TableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Объект", "Количество"])
        layout.addWidget(self.table)
        
    def update_results(self, results):
        """Обновляет результаты в таблице"""
        self.table.setRowCount(len(results))
        for i, result in enumerate(results):
            self.table.setItem(i, 0, QTableWidgetItem(result.get('class', '')))
            self.table.setItem(i, 1, QTableWidgetItem(str(result.get('count', 0))))

class CoordinateInput(QWidget):
    """Виджет для ввода координат"""
    
    coordinates_changed = pyqtSignal(float, float, int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QGridLayout(self)
        
        # Широта
        layout.addWidget(QLabel("Широта:"), 0, 0)
        self.lat_input = QLineEdit("55.82103")
        self.lat_input.textChanged.connect(self.on_coordinates_changed)
        layout.addWidget(self.lat_input, 0, 1)
        
        # Долгота
        layout.addWidget(QLabel("Долгота:"), 1, 0)
        self.lon_input = QLineEdit("49.16219")
        self.lon_input.textChanged.connect(self.on_coordinates_changed)
        layout.addWidget(self.lon_input, 1, 1)
        
        # Масштаб
        layout.addWidget(QLabel("Масштаб:"), 2, 0)
        self.scale_input = QSpinBox()
        self.scale_input.setRange(1, 20)
        self.scale_input.setValue(16)
        self.scale_input.valueChanged.connect(self.on_coordinates_changed)
        layout.addWidget(self.scale_input, 2, 1)
        
    def on_coordinates_changed(self):
        """Обработка изменения координат"""
        try:
            lat = float(self.lat_input.text())
            lon = float(self.lon_input.text())
            scale = self.scale_input.value()
            self.coordinates_changed.emit(lat, lon, scale)
        except ValueError:
            pass
            
    def get_coordinates(self):
        """Возвращает текущие координаты"""
        try:
            lat = float(self.lat_input.text())
            lon = float(self.lon_input.text())
            scale = self.scale_input.value()
            return lat, lon, scale
        except ValueError:
            return None, None, None
