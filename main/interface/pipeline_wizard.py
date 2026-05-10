"""Modal wizard for the «Полный SAR пайплайн» button.

The wizard walks the user through two interactive stages on a single image:

1. Image enhancement (algorithm + intensity, applied via ``EnhancementWorker``).
2. ROI detection (type / sensitivity / brightness range, via ``ROIWorker``).

For each stage the user may press «Применить» multiple times — the latest
result overrides previous output. Pressing «Далее» without applying anything
forwards the current input to the next stage unchanged. After the second stage
the dialog returns the final image path and the list of detected regions; the
controller then runs YOLOv11m inference and renders results on the main page.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from PyQt5.QtCore import QRectF, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QColor, QPen
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QFrame,
    QGraphicsRectItem,
    QGraphicsView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpacerItem,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .view import MyGraphicsView


class _RoiEditableGraphicsView(MyGraphicsView):
    """Подкласс ``MyGraphicsView`` с режимом ручного рисования прямоугольников.

    В обычном режиме поведение унаследовано от родителя (ПКМ-пан, колесо-зум).
    В draw-режиме ЛКМ-drag создаёт временную «черновую» рамку на сцене; на
    release нормализованные координаты сцены (которые 1:1 совпадают с
    пиксельными координатами изображения) эмитятся сигналом ``region_drawn``,
    и режим автоматически выключается.
    """

    region_drawn = pyqtSignal(int, int, int, int)
    draw_mode_changed = pyqtSignal(bool)

    _MIN_BOX_SIZE = 6  # px по каждой стороне; меньшее молча игнорируем

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._draw_mode: bool = False
        self._drawing: bool = False
        self._draw_start = None  # QPointF в координатах сцены или None
        self._rubber: Optional[QGraphicsRectItem] = None

    def set_draw_mode(self, enabled: bool) -> None:
        if enabled == self._draw_mode:
            return
        self._draw_mode = enabled
        if enabled:
            self.setDragMode(QGraphicsView.NoDrag)
            self.setCursor(Qt.CrossCursor)
            self.setFocus(Qt.OtherFocusReason)
        else:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.setCursor(Qt.ArrowCursor)
        self._cleanup_rubber()
        self.draw_mode_changed.emit(enabled)

    def _cleanup_rubber(self) -> None:
        if self._rubber is not None:
            try:
                self.scene.removeItem(self._rubber)
            except Exception:
                pass
            self._rubber = None
        self._drawing = False
        self._draw_start = None

    def _make_rubber(self, rect: QRectF) -> QGraphicsRectItem:
        item = QGraphicsRectItem(rect)
        pen = QPen(QColor(0, 255, 0))
        pen.setWidth(2)
        pen.setCosmetic(True)
        item.setPen(pen)
        return item

    def mousePressEvent(self, event):  # type: ignore[override]
        if self._draw_mode and event.button() == Qt.LeftButton:
            start = self.mapToScene(event.pos())
            self._draw_start = start
            self._drawing = True
            self._rubber = self._make_rubber(QRectF(start, start))
            self.scene.addItem(self._rubber)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # type: ignore[override]
        if self._drawing and self._rubber is not None and self._draw_start is not None:
            cur = self.mapToScene(event.pos())
            rect = QRectF(self._draw_start, cur).normalized()
            self._rubber.setRect(rect)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # type: ignore[override]
        if self._drawing and event.button() == Qt.LeftButton:
            cur = self.mapToScene(event.pos())
            start = self._draw_start
            self._cleanup_rubber()
            self.set_draw_mode(False)
            event.accept()
            if start is None:
                return
            scene_rect = self.scene.sceneRect()
            x1 = int(round(min(start.x(), cur.x())))
            y1 = int(round(min(start.y(), cur.y())))
            x2 = int(round(max(start.x(), cur.x())))
            y2 = int(round(max(start.y(), cur.y())))
            x1 = max(0, min(int(scene_rect.width()), x1))
            y1 = max(0, min(int(scene_rect.height()), y1))
            x2 = max(0, min(int(scene_rect.width()), x2))
            y2 = max(0, min(int(scene_rect.height()), y2))
            if (x2 - x1) >= self._MIN_BOX_SIZE and (y2 - y1) >= self._MIN_BOX_SIZE:
                self.region_drawn.emit(x1, y1, x2, y2)
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):  # type: ignore[override]
        if self._draw_mode and event.key() == Qt.Key_Escape:
            self._cleanup_rubber()
            self.set_draw_mode(False)
            event.accept()
            return
        super().keyPressEvent(event)


_STAGE_ENHANCE = 0
_STAGE_ROI = 1
_STAGE_COUNT = 2


def _project_tmp_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", "tmp"))


class _StagePageBase(QWidget):
    """Common scaffolding for a wizard stage page.

    Provides a 2-column layout (settings on the left, preview on the right)
    and helpers for setting the preview image / status messages. Subclasses
    fill in stage-specific controls and worker creation.
    """

    title: str = ""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._worker = None  # type: Optional[Any]
        self._setup_layout()

    def _setup_layout(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)

        self.controls_panel = QFrame()
        self.controls_panel.setFrameStyle(QFrame.StyledPanel)
        self.controls_panel.setMaximumWidth(360)
        self.controls_panel.setMinimumWidth(320)
        self.controls_layout = QVBoxLayout(self.controls_panel)
        self.controls_layout.setContentsMargins(8, 8, 8, 8)
        root.addWidget(self.controls_panel)

        self.preview = self._make_preview()
        self.preview.setMinimumSize(560, 480)
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self.preview, stretch=1)

    def _make_preview(self) -> MyGraphicsView:
        """Фабрика виджета превью; подклассы могут вернуть кастомный класс."""
        return MyGraphicsView()

    def show_image(self, path: Optional[str]) -> None:
        if path and os.path.isfile(path):
            self.preview.set_image(path)

    def stop_worker(self) -> None:
        """Cleanly stop a running worker, if any."""
        worker = self._worker
        if worker is None:
            return
        try:
            worker.progress.disconnect()
        except Exception:
            pass
        try:
            worker.finished.disconnect()
        except Exception:
            pass
        try:
            worker.error.disconnect()
        except Exception:
            pass
        try:
            if worker.isRunning():
                worker.requestInterruption()
                worker.wait(50)
        except Exception:
            pass
        self._worker = None


class _EnhanceStagePage(_StagePageBase):
    """Stage 1: SAR image enhancement (combo + intensity slider + info)."""

    title = "Этап 1 из 2 · Улучшение качества SAR снимка"

    _PARAM_INFO_BY_TYPE = {
        "Гибридное подавление шума SAR": (
            "Гибридное подавление шума SAR\n"
            "• Алгоритм: bilateral filter + Non-local Means.\n"
            "• Подходит для умеренного спекла без сильной потери деталей.\n"
        ),
        "Адаптивное подавление спекла SAR": (
            "Адаптивное подавление спекла SAR\n"
            "• Алгоритм: адаптивный фильтр в духе Lee/Frost.\n"
            "• Сильное подавление зернистого спекла, сохраняет границы объектов.\n"
        ),
        "Анизотропная диффузия SAR": (
            "Анизотропная диффузия SAR\n"
            "• Алгоритм: Perona–Malik / SRAD.\n"
            "• Хорошо сглаживает однородные участки и сохраняет края.\n"
        ),
    }

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        group = QGroupBox("Настройки улучшения SAR снимков")
        group_layout = QVBoxLayout(group)

        group_layout.addWidget(QLabel("Тип улучшения:"))
        self.enhance_type_combo = QComboBox()
        self.enhance_type_combo.addItems(list(self._PARAM_INFO_BY_TYPE.keys()))
        group_layout.addWidget(self.enhance_type_combo)

        group_layout.addWidget(QLabel("Интенсивность:"))
        self.intensity_slider = QSlider(Qt.Horizontal)
        self.intensity_slider.setRange(1, 100)
        self.intensity_slider.setValue(50)
        group_layout.addWidget(self.intensity_slider)

        self.intensity_label = QLabel("Интенсивность: 50%")
        self.intensity_label.setStyleSheet("color: #cccccc; font-size: 12px;")
        group_layout.addWidget(self.intensity_label)

        self.controls_layout.addWidget(group)

        info_group = QGroupBox("Информация о параметрах")
        info_layout = QVBoxLayout(info_group)
        self.param_info = QTextEdit()
        self.param_info.setReadOnly(True)
        self.param_info.setMaximumHeight(140)
        self.param_info.setStyleSheet(
            "QTextEdit { background-color: #404040; color: #ffffff;"
            " border: 1px solid #666666; font-size: 11px; }"
        )
        info_layout.addWidget(self.param_info)
        self.controls_layout.addWidget(info_group)

        self.metrics_label = QLabel("Метрики появятся после применения.")
        self.metrics_label.setWordWrap(True)
        self.metrics_label.setStyleSheet("color: #cccccc; font-size: 11px;")
        self.controls_layout.addWidget(self.metrics_label)

        self.controls_layout.addStretch()

        self.intensity_slider.valueChanged.connect(self._on_intensity_changed)
        self.enhance_type_combo.currentTextChanged.connect(self._on_type_changed)
        self._on_type_changed(self.enhance_type_combo.currentText())

    def _on_intensity_changed(self, value: int) -> None:
        self.intensity_label.setText(f"Интенсивность: {int(value)}%")

    def _on_type_changed(self, text: str) -> None:
        self.param_info.setText(
            self._PARAM_INFO_BY_TYPE.get(text, "Выберите тип улучшения для подробностей.")
        )

    def get_settings(self) -> Dict[str, Any]:
        return {
            "enhance_type": self.enhance_type_combo.currentText(),
            "intensity": int(self.intensity_slider.value()),
        }

    def make_worker(self, image_path: str):
        # Lazy import to avoid circular dependency with new_controller.
        from .new_controller import EnhancementWorker

        s = self.get_settings()
        self._worker = EnhancementWorker(
            image_path=image_path,
            enhance_type=s["enhance_type"],
            intensity=s["intensity"],
        )
        return self._worker

    def show_metrics(self, metrics: Optional[Dict[str, Any]]) -> None:
        if not metrics:
            self.metrics_label.setText("Метрики недоступны для этого результата.")
            return
        self.metrics_label.setText(
            f"PSNR: {metrics.get('psnr', 0):.2f} dB · "
            f"Контраст: +{metrics.get('contrast_improvement', 0):.1f}% · "
            f"Яркость: {metrics.get('brightness_change', 0):+.1f}%"
        )

    def reset_metrics(self) -> None:
        self.metrics_label.setText("Метрики появятся после применения.")


class _RoiStagePage(_StagePageBase):
    """Stage 2: ROI detection (type / sensitivity / brightness range)."""

    title = "Этап 2 из 2 · Зоны интересов"

    # Сигнал, что пользователь отредактировал регионы (например, удалил один из них
    # или добавил вручную). Визард слушает его, чтобы синхронизировать stage_outputs
    # и итоговое превью.
    regions_changed = pyqtSignal()

    def _make_preview(self) -> MyGraphicsView:
        return _RoiEditableGraphicsView()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._regions: List[Dict[str, Any]] = []
        self._image_with_boxes: Optional[str] = None
        # Базовое (без рамок) изображение, по которому считались ROI; нужно,
        # чтобы перерисовывать превью при ручном удалении/добавлении регионов.
        self._base_image_path: Optional[str] = None
        # Кеш grayscale-копии базового изображения для быстрого расчёта яркости
        # при добавлении ручных регионов.
        self._base_gray = None  # numpy.ndarray | None
        self._selected_region_index: Optional[int] = None

        # Сигналы редактируемого превью пробрасываем в обработчики страницы.
        if isinstance(self.preview, _RoiEditableGraphicsView):
            self.preview.region_drawn.connect(self._on_region_drawn)
            self.preview.draw_mode_changed.connect(self._on_draw_mode_changed)

        group = QGroupBox("Настройки зон интересов")
        group_layout = QVBoxLayout(group)

        group_layout.addWidget(QLabel("Тип анализа:"))
        self.roi_type_combo = QComboBox()
        self.roi_type_combo.addItems([
            "Автоматическое выделение",
            "Ручное выделение",
        ])
        group_layout.addWidget(self.roi_type_combo)

        group_layout.addWidget(QLabel("Чувствительность:"))
        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setRange(1, 100)
        self.sensitivity_slider.setValue(50)
        group_layout.addWidget(self.sensitivity_slider)

        group_layout.addWidget(QLabel("Яркость областей (мин/макс):"))
        br_layout = QGridLayout()
        br_layout.addWidget(QLabel("Мин:"), 0, 0)
        self.bright_min_slider = QSlider(Qt.Horizontal)
        self.bright_min_slider.setRange(0, 255)
        self.bright_min_slider.setValue(0)
        self.bright_min_label = QLabel("0")
        br_layout.addWidget(self.bright_min_slider, 0, 1)
        br_layout.addWidget(self.bright_min_label, 0, 2)

        br_layout.addWidget(QLabel("Макс:"), 1, 0)
        self.bright_max_slider = QSlider(Qt.Horizontal)
        self.bright_max_slider.setRange(0, 255)
        self.bright_max_slider.setValue(255)
        self.bright_max_label = QLabel("255")
        br_layout.addWidget(self.bright_max_slider, 1, 1)
        br_layout.addWidget(self.bright_max_label, 1, 2)
        group_layout.addLayout(br_layout)

        self.bright_min_slider.valueChanged.connect(self._on_brightness_changed)
        self.bright_max_slider.valueChanged.connect(self._on_brightness_changed)

        self.controls_layout.addWidget(group)

        results_group = QGroupBox("Результаты анализа")
        results_layout = QVBoxLayout(results_group)

        self.results_count_label = QLabel("Найдено областей: —")
        self.results_count_label.setStyleSheet("color: #cccccc;")
        results_layout.addWidget(self.results_count_label)

        self.results_hint_label = QLabel(
            "Клик — подсветить область, двойной клик — удалить её."
        )
        self.results_hint_label.setStyleSheet("color: #888888; font-size: 11px;")
        self.results_hint_label.setWordWrap(True)
        results_layout.addWidget(self.results_hint_label)

        self.add_region_btn = QPushButton("Добавить область")
        self.add_region_btn.setCheckable(True)
        self.add_region_btn.setEnabled(False)
        self.add_region_btn.setToolTip(
            "Левая кнопка мыши — нарисовать рамку. Esc — отмена. "
            "После одной рамки режим выключится автоматически."
        )
        self.add_region_btn.toggled.connect(self._on_add_region_toggled)
        results_layout.addWidget(self.add_region_btn)

        self.results_list = QListWidget()
        self.results_list.setMinimumHeight(160)
        self.results_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.results_list.itemClicked.connect(self._on_item_clicked)
        self.results_list.itemDoubleClicked.connect(self._on_item_double_clicked)
        results_layout.addWidget(self.results_list)

        self.controls_layout.addWidget(results_group)

        self.controls_layout.addStretch()

    def _on_brightness_changed(self, _value: int) -> None:
        mn = self.bright_min_slider.value()
        mx = self.bright_max_slider.value()
        if mn > mx:
            self.bright_min_slider.blockSignals(True)
            self.bright_min_slider.setValue(mx)
            self.bright_min_slider.blockSignals(False)
            mn = mx
        self.bright_min_label.setText(str(mn))
        self.bright_max_label.setText(str(mx))

    def get_settings(self) -> Dict[str, Any]:
        return {
            "roi_type": self.roi_type_combo.currentText(),
            "sensitivity": int(self.sensitivity_slider.value()),
            "bright_min": int(self.bright_min_slider.value()),
            "bright_max": int(self.bright_max_slider.value()),
        }

    def make_worker(self, image_path: str):
        from .new_controller import ROIWorker

        s = self.get_settings()
        self._worker = ROIWorker(
            image_path=image_path,
            roi_type=s["roi_type"],
            sensitivity=s["sensitivity"],
            bright_min=s["bright_min"],
            bright_max=s["bright_max"],
        )
        return self._worker

    def set_base_image(self, path: Optional[str]) -> None:
        """Запомнить исходник этапа — нужен для перерисовки при ручных правках.

        Дополнительно кеширует grayscale-копию изображения для расчёта яркости
        ручных регионов и включает/выключает кнопку «Добавить область».
        """
        self._base_image_path = path
        self._base_gray = None
        valid = bool(path) and os.path.isfile(path or "")
        if valid:
            try:
                import cv2  # ленивый импорт
                gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if gray is not None:
                    self._base_gray = gray
            except Exception:
                self._base_gray = None
        if hasattr(self, "add_region_btn"):
            self.add_region_btn.setEnabled(valid)

    def populate_results(
        self,
        results: Dict[str, Any],
        base_image_path: Optional[str] = None,
    ) -> None:
        # Любая активная сессия рисования сбрасывается, чтобы кнопка не залипала.
        if isinstance(self.preview, _RoiEditableGraphicsView):
            self.preview.set_draw_mode(False)

        self._regions = list(results.get("regions", []))
        self._image_with_boxes = results.get("image_with_boxes")
        if base_image_path is not None:
            self.set_base_image(base_image_path)
        self._selected_region_index = None
        self._refresh_list_widget()

    def reset_results(self) -> None:
        if isinstance(self.preview, _RoiEditableGraphicsView):
            self.preview.set_draw_mode(False)
        self._regions = []
        self._image_with_boxes = None
        self._selected_region_index = None
        self.results_list.clear()
        self.results_count_label.setText("Найдено областей: —")

    def _refresh_list_widget(self) -> None:
        """Перезаполнить список регионов; индекс региона хранится в UserRole."""
        self.results_list.blockSignals(True)
        self.results_list.clear()
        for idx, r in enumerate(self._regions):
            text = (
                f"({r['x1']},{r['y1']})-({r['x2']},{r['y2']}): "
                f"ярк. {r.get('brightness', 0):.1f}"
            )
            if r.get("source") == "manual":
                text += " (вручную)"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, idx)
            self.results_list.addItem(item)
        self.results_list.blockSignals(False)
        self.results_count_label.setText(
            f"Найдено областей: {len(self._regions)}"
        )
        if (
            self._selected_region_index is not None
            and 0 <= self._selected_region_index < self.results_list.count()
        ):
            self.results_list.setCurrentRow(self._selected_region_index)

    def _render_with_regions(self, highlight_index: Optional[int]) -> Optional[str]:
        """Перерисовать рамки на исходнике этапа и вернуть путь к новому файлу.

        Цвета: выделенный — красный, ручные — зелёные, авто — жёлтые.
        """
        if not self._base_image_path or not os.path.isfile(self._base_image_path):
            return None
        try:
            import cv2  # lazy import — модуль визарда не зависит от cv2 напрямую
        except Exception:
            return None

        img = cv2.imread(self._base_image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        vis = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for i, r in enumerate(self._regions):
            try:
                x1 = int(r["x1"]); y1 = int(r["y1"])
                x2 = int(r["x2"]); y2 = int(r["y2"])
            except Exception:
                continue
            mb = float(r.get("brightness", 0.0))
            if highlight_index is not None and i == highlight_index:
                color = (0, 0, 255)  # красный
                thickness = 3
            elif r.get("source") == "manual":
                color = (0, 255, 0)  # зелёный для ручных
                thickness = 2
            else:
                color = (0, 255, 255)  # жёлтый для авто
                thickness = 2
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(
                vis, f"{mb:.1f}", (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
            )

        out_dir = _project_tmp_dir()
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(self._base_image_path))[0]
        out_path = os.path.join(out_dir, f"{base}_roi_edit.jpg")
        try:
            cv2.imwrite(out_path, vis)
        except Exception:
            return None
        return out_path

    def _on_add_region_toggled(self, checked: bool) -> None:
        """Слот клика по toggle-кнопке: переводим превью в режим рисования."""
        if not isinstance(self.preview, _RoiEditableGraphicsView):
            return
        # Если включают режим без валидного исходника — отжимаем кнопку обратно.
        if checked and not (self._base_image_path and os.path.isfile(self._base_image_path)):
            self.add_region_btn.blockSignals(True)
            self.add_region_btn.setChecked(False)
            self.add_region_btn.blockSignals(False)
            return
        self.preview.set_draw_mode(checked)

    def _on_draw_mode_changed(self, enabled: bool) -> None:
        """Синхронизирует состояние кнопки при авто-выходе/Esc из режима."""
        if self.add_region_btn.isChecked() == enabled:
            return
        self.add_region_btn.blockSignals(True)
        self.add_region_btn.setChecked(enabled)
        self.add_region_btn.blockSignals(False)

    def _on_region_drawn(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Добавить нарисованный регион в список и обновить превью."""
        if not (self._base_image_path and os.path.isfile(self._base_image_path)):
            return
        bm = 0.0
        if self._base_gray is not None:
            try:
                patch = self._base_gray[y1:y2, x1:x2]
                if patch.size:
                    bm = float(patch.mean())
            except Exception:
                bm = 0.0
        self._regions.append({
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
            "brightness": bm,
            "source": "manual",
        })
        self._selected_region_index = None
        self._refresh_list_widget()
        new_path = self._render_with_regions(None)
        if new_path:
            self._image_with_boxes = new_path
            self.show_image(new_path)
        self.regions_changed.emit()

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        idx = item.data(Qt.UserRole)
        if not isinstance(idx, int) or not (0 <= idx < len(self._regions)):
            return
        self._selected_region_index = idx
        new_path = self._render_with_regions(idx)
        if new_path:
            self._image_with_boxes = new_path
            self.show_image(new_path)
            # Сообщаем визарду, чтобы он подменил stage_outputs текущим превью —
            # тогда при возврате на этот шаг будет видна актуальная картинка.
            self.regions_changed.emit()

    def _on_item_double_clicked(self, item: QListWidgetItem) -> None:
        idx = item.data(Qt.UserRole)
        if not isinstance(idx, int) or not (0 <= idx < len(self._regions)):
            return
        r = self._regions[idx]
        text = (
            f"Удалить область ({r['x1']},{r['y1']})-({r['x2']},{r['y2']}), "
            f"яркость {r.get('brightness', 0):.1f}?"
        )
        btn = QMessageBox.question(
            self,
            "Удалить область",
            text,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if btn != QMessageBox.Yes:
            return

        del self._regions[idx]
        self._selected_region_index = None
        self._refresh_list_widget()

        new_path = self._render_with_regions(None)
        if new_path:
            self._image_with_boxes = new_path
            self.show_image(new_path)
        self.regions_changed.emit()

    @property
    def regions(self) -> List[Dict[str, Any]]:
        return list(self._regions)

    @property
    def image_with_boxes(self) -> Optional[str]:
        return self._image_with_boxes


class PipelineWizardDialog(QDialog):
    """Two-stage modal pipeline wizard. See module docstring."""

    def __init__(self, image_path: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Полный SAR пайплайн")
        self.setModal(True)
        self.resize(1100, 760)

        self._initial_image_path: str = image_path
        self._stage_inputs: List[str] = [image_path, image_path]
        self._stage_outputs: List[Optional[str]] = [None, None]
        self._current_stage: int = 0
        self._busy: bool = False

        self._final_image_path: Optional[str] = image_path
        self._final_regions: List[Dict[str, Any]] = []

        self._build_ui()
        self._refresh_ui_for_current_stage()

    # ---------- public results ----------

    @property
    def final_image_path(self) -> Optional[str]:
        return self._final_image_path

    @property
    def final_regions(self) -> List[Dict[str, Any]]:
        return list(self._final_regions)

    # ---------- UI scaffolding ----------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        self.header_label = QLabel()
        f = self.header_label.font()
        f.setPointSize(f.pointSize() + 1)
        f.setBold(True)
        self.header_label.setFont(f)
        root.addWidget(self.header_label)

        self.stack = QStackedWidget()
        self.enhance_page = _EnhanceStagePage(self)
        self.roi_page = _RoiStagePage(self)
        self.stack.addWidget(self.enhance_page)
        self.stack.addWidget(self.roi_page)
        root.addWidget(self.stack, stretch=1)

        # Status / progress strip
        status_row = QHBoxLayout()
        self.status_label = QLabel("Готов к применению.")
        self.status_label.setStyleSheet("color: #cccccc;")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(260)
        status_row.addWidget(self.status_label, stretch=1)
        status_row.addWidget(self.progress_bar)
        root.addLayout(status_row)

        # Bottom navigation row
        nav = QHBoxLayout()
        self.back_btn = QPushButton("Назад")
        self.reset_stage_btn = QPushButton("Сбросить этап")
        self.restart_btn = QPushButton("Начать сначала")
        self.apply_btn = QPushButton("Применить")
        self.next_btn = QPushButton("Далее")
        self.cancel_btn = QPushButton("Отмена")

        nav.addWidget(self.back_btn)
        nav.addWidget(self.reset_stage_btn)
        nav.addWidget(self.restart_btn)
        nav.addItem(QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        nav.addWidget(self.apply_btn)
        nav.addWidget(self.next_btn)
        nav.addWidget(self.cancel_btn)
        root.addLayout(nav)

        self.back_btn.clicked.connect(self._go_back)
        self.reset_stage_btn.clicked.connect(self._reset_current_stage)
        self.restart_btn.clicked.connect(self._restart)
        self.apply_btn.clicked.connect(self._apply_current_stage)
        self.next_btn.clicked.connect(self._go_next)
        self.cancel_btn.clicked.connect(self.reject)

        # Синхронизация stage_outputs при ручном редактировании регионов
        # (одиночный клик подсвечивает регион, двойной — удаляет).
        self.roi_page.regions_changed.connect(self._on_roi_regions_changed)

    # ---------- state helpers ----------

    def _current_input_path(self) -> str:
        return self._stage_inputs[self._current_stage]

    def _current_effective_path(self) -> str:
        """The path used for preview / forwarding: latest output if any, else input."""
        out = self._stage_outputs[self._current_stage]
        return out if out else self._stage_inputs[self._current_stage]

    def _refresh_ui_for_current_stage(self) -> None:
        page: _StagePageBase = self.stack.widget(self._current_stage)  # type: ignore[assignment]
        self.stack.setCurrentIndex(self._current_stage)

        self.header_label.setText(page.title)

        # Preview shows the most recent output for this stage (or stage input).
        page.show_image(self._current_effective_path())

        # На входе в ROI-этап даём странице знать, какое именно изображение
        # использовать для перерисовки рамок и расчёта яркости при ручных правках.
        # Это также включает кнопку «Добавить область», даже если пользователь
        # ещё ни разу не нажимал «Применить».
        if self._current_stage == _STAGE_ROI:
            self.roi_page.set_base_image(self._stage_inputs[_STAGE_ROI])

        # Progress / status
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Готов к применению.")

        # Buttons
        is_last = self._current_stage == _STAGE_COUNT - 1
        self.back_btn.setEnabled(self._current_stage > 0 and not self._busy)
        self.reset_stage_btn.setEnabled(not self._busy)
        self.restart_btn.setEnabled(not self._busy)
        self.apply_btn.setEnabled(not self._busy)
        self.next_btn.setEnabled(not self._busy)
        self.cancel_btn.setEnabled(not self._busy)
        self.next_btn.setText("Готово" if is_last else "Далее")

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        for btn in (
            self.back_btn,
            self.reset_stage_btn,
            self.restart_btn,
            self.apply_btn,
            self.next_btn,
            self.cancel_btn,
        ):
            btn.setEnabled(not busy)
        # Кнопка ручного добавления области — только во время простоя ROI-этапа.
        if hasattr(self, "roi_page") and hasattr(self.roi_page, "add_region_btn"):
            if busy:
                # При запуске воркера выходим из draw-режима и блокируем кнопку.
                if isinstance(self.roi_page.preview, _RoiEditableGraphicsView):
                    self.roi_page.preview.set_draw_mode(False)
                self.roi_page.add_region_btn.setEnabled(False)
            else:
                base = self.roi_page._base_image_path
                self.roi_page.add_region_btn.setEnabled(
                    bool(base) and os.path.isfile(base or "")
                )
        if busy:
            self.progress_bar.setVisible(True)
        else:
            # Скрываем бар по завершении этапа и восстанавливаем доступность
            # кнопки «Назад» по индексу страницы.
            self.progress_bar.setVisible(False)
            self.progress_bar.setValue(0)
            self.back_btn.setEnabled(self._current_stage > 0)

    # ---------- actions ----------

    def _go_back(self) -> None:
        if self._busy or self._current_stage == 0:
            return
        # Stop any in-flight worker on the current page before leaving it.
        page: _StagePageBase = self.stack.widget(self._current_stage)  # type: ignore[assignment]
        page.stop_worker()
        self._current_stage -= 1
        self._refresh_ui_for_current_stage()

    def _go_next(self) -> None:
        if self._busy:
            return

        # If user did not apply anything on this stage, the input is forwarded as-is.
        forwarded = self._current_effective_path()

        if self._current_stage == _STAGE_COUNT - 1:
            self._final_image_path = forwarded
            roi_page: _RoiStagePage = self.roi_page
            self._final_regions = roi_page.regions
            if not self._final_regions:
                btn = QMessageBox.question(
                    self,
                    "Зоны не найдены",
                    (
                        "Вы не применяли анализ зон или зон не найдено. "
                        "Закрыть мастер без запуска YOLOv11m?"
                    ),
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if btn != QMessageBox.Yes:
                    return
            self.accept()
            return

        # Move to next stage; its input becomes the effective output of this stage.
        self._stage_inputs[self._current_stage + 1] = forwarded
        self._current_stage += 1
        self._refresh_ui_for_current_stage()

    def _reset_current_stage(self) -> None:
        if self._busy:
            return
        page: _StagePageBase = self.stack.widget(self._current_stage)  # type: ignore[assignment]
        page.stop_worker()
        self._stage_outputs[self._current_stage] = None
        if isinstance(page, _RoiStagePage):
            page.reset_results()
        if isinstance(page, _EnhanceStagePage):
            page.reset_metrics()
        self._refresh_ui_for_current_stage()

    def _restart(self) -> None:
        if self._busy:
            return
        # Stop workers everywhere
        for i in range(_STAGE_COUNT):
            page: _StagePageBase = self.stack.widget(i)  # type: ignore[assignment]
            page.stop_worker()
        self._stage_outputs = [None, None]
        self._stage_inputs = [self._initial_image_path, self._initial_image_path]
        self.roi_page.reset_results()
        self.enhance_page.reset_metrics()
        self._current_stage = 0
        self._refresh_ui_for_current_stage()

    def _apply_current_stage(self) -> None:
        if self._busy:
            return
        page: _StagePageBase = self.stack.widget(self._current_stage)  # type: ignore[assignment]
        page.stop_worker()

        try:
            worker = page.make_worker(self._stage_inputs[self._current_stage])
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка", f"Не удалось подготовить этап: {exc}")
            return

        worker.progress.connect(self._on_worker_progress)
        if isinstance(page, _EnhanceStagePage):
            worker.finished.connect(self._on_enhance_finished)
        else:
            worker.finished.connect(self._on_roi_finished)
        worker.error.connect(self._on_worker_error)

        self._set_busy(True)
        self.status_label.setText("Запуск...")
        self.progress_bar.setValue(0)
        worker.start()

    # ---------- worker callbacks ----------

    @pyqtSlot(int, str)
    def _on_worker_progress(self, percent: int, message: str) -> None:
        self.progress_bar.setValue(int(max(0, min(100, percent))))
        if message:
            self.status_label.setText(message)

    @pyqtSlot(dict)
    def _on_enhance_finished(self, results: Dict[str, Any]) -> None:
        self._set_busy(False)
        path = results.get("enhanced_image")
        if not path or not os.path.isfile(path):
            QMessageBox.critical(self, "Ошибка", "Улучшение не вернуло пригодный файл.")
            self.status_label.setText("Не удалось получить результат улучшения.")
            return

        self._stage_outputs[_STAGE_ENHANCE] = path
        self.enhance_page.show_image(path)
        self.enhance_page.show_metrics(results.get("quality_metrics"))
        self.status_label.setText("Улучшение применено. Можно нажать «Далее» или подобрать другие параметры.")

    @pyqtSlot(dict)
    def _on_roi_finished(self, results: Dict[str, Any]) -> None:
        self._set_busy(False)
        # Базовое (без рамок) изображение — это вход этого этапа; нужен,
        # чтобы можно было перерисовать рамки при удалении области руками.
        self.roi_page.populate_results(
            results,
            base_image_path=self._stage_inputs[_STAGE_ROI],
        )
        preview = results.get("image_with_boxes") or self._stage_inputs[_STAGE_ROI]
        if preview and os.path.isfile(preview):
            self.roi_page.show_image(preview)
            # Сохраняем превью с боксами как «выход» этапа — оно показывается
            # после возврата на этот шаг или при «Сбросить этап».
            self._stage_outputs[_STAGE_ROI] = preview
        cnt = int(results.get("count", len(self.roi_page.regions)))
        if cnt == 0:
            self.status_label.setText("Зон не найдено. Поправьте параметры и попробуйте снова.")
        else:
            self.status_label.setText(
                f"Найдено зон: {cnt}. Кликните по строке, чтобы подсветить область, "
                f"двойной клик — удалить. «Готово» запустит YOLOv11m."
            )

    def _on_roi_regions_changed(self) -> None:
        """Обновляем выход этапа, когда пользователь редактирует регионы."""
        preview = self.roi_page.image_with_boxes
        if preview and os.path.isfile(preview):
            self._stage_outputs[_STAGE_ROI] = preview
        cnt = len(self.roi_page.regions)
        self.status_label.setText(
            f"Областей после правок: {cnt}. Можно продолжать редактирование "
            f"или нажать «Готово»."
        )

    @pyqtSlot(str)
    def _on_worker_error(self, message: str) -> None:
        self._set_busy(False)
        self.status_label.setText(f"Ошибка: {message}")
        QMessageBox.critical(self, "Ошибка пайплайна", message)

    # ---------- lifecycle ----------

    def reject(self) -> None:  # type: ignore[override]
        # Stop workers if user cancels mid-run.
        for i in range(_STAGE_COUNT):
            page: _StagePageBase = self.stack.widget(i)  # type: ignore[assignment]
            page.stop_worker()
        super().reject()


__all__ = ["PipelineWizardDialog"]
