"""
Компоненты отображения для интерфейса
"""

import os
from PyQt5.QtWidgets import (QTableWidget, QTableWidgetItem, QGraphicsView, 
                             QGraphicsScene, QGraphicsPixmapItem)
from PyQt5.QtCore import Qt, pyqtSignal, QRectF, QPointF
from PyQt5.QtGui import QPixmap, QWheelEvent, QMouseEvent


class TableWidget(QTableWidget):
    """Таблица для отображения результатов"""
    image_changes = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cellDoubleClicked.connect(self.on_cell_double_clicked)
        self.setEditTriggers(QTableWidget.NoEditTriggers)

    def fill_table(self, file_names):
        """Заполняет таблицу именами файлов"""
        for file_name in file_names:
            self.add_row(file_name)
            
    def add_row(self, file_name, value="Не определено"):
        """Добавляет строку в таблицу"""
        row = self.rowCount()
        self.insertRow(row)
        self.setItem(row, 0, QTableWidgetItem(file_name))
        self.setItem(row, 1, QTableWidgetItem(str(value)))
        self.item(row, 0).setFlags(Qt.ItemIsEnabled)
        self.item(row, 1).setFlags(Qt.ItemIsEnabled)

    def update_value(self, file_name, value):
        """Обновляет значение в таблице"""
        row_count = self.rowCount()
        for i in range(row_count):
            current_file_name = self.item(i, 0).text()
            if current_file_name == file_name:
                self.setItem(i, 1, QTableWidgetItem(str(value)))
                return

    def on_cell_double_clicked(self, row, col):
        """Обработка двойного клика по ячейке"""
        current_filename = self.item(row, 0).text()
        self.image_changes.emit(current_filename)


class MyGraphicsView(QGraphicsView):
    """Виджет для отображения изображений с возможностью масштабирования и панорамирования"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        self._pan_start = QPointF()
        self._panning = False
        
        # Настройки масштабирования
        self.zoom_factor = 1.25
        self.min_zoom = 0.1
        self.max_zoom = 20.0
        self.current_scale = 1.0

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
        self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
        self.current_scale = 1.0

    def wheelEvent(self, event: QWheelEvent):
        """Обработка масштабирования колесиком мыши"""
        # Получаем позицию мыши относительно сцены
        mouse_pos = event.pos()
        scene_pos = self.mapToScene(mouse_pos)
        
        # Определяем направление масштабирования
        zoom_in = event.angleDelta().y() > 0
        
        if zoom_in and self.current_scale < self.max_zoom:
            # Масштабируем к позиции мыши
            self.scale(self.zoom_factor, self.zoom_factor)
            self.current_scale *= self.zoom_factor
            
            # Центрируем на позиции мыши
            new_pos = self.mapToScene(mouse_pos)
            delta = new_pos - scene_pos
            self.translate(delta.x(), delta.y())
            
        elif not zoom_in and self.current_scale > self.min_zoom:
            # Масштабируем от позиции мыши
            self.scale(1 / self.zoom_factor, 1 / self.zoom_factor)
            self.current_scale /= self.zoom_factor
            
            # Центрируем на позиции мыши
            new_pos = self.mapToScene(mouse_pos)
            delta = new_pos - scene_pos
            self.translate(delta.x(), delta.y())

    def mousePressEvent(self, event: QMouseEvent):
        """Обработка нажатий мыши"""
        if event.button() == Qt.RightButton:
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
        if event.button() == Qt.RightButton:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
        else:
            super().mouseReleaseEvent(event)


