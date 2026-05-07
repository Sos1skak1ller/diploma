import os
import sys
from typing import Optional

from PyQt5.QtWidgets import QFileDialog, QMessageBox, QListWidgetItem, QTableWidgetItem
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QPixmap
import cv2
import numpy as np
import yaml

# SAR pipeline imports (путь к локальному пакету внутри main.algorythms)
try:
    from main.algorythms.area_selection.sar_gate_yolo.utils import read_yaml
    from main.algorythms.area_selection.sar_gate_yolo.cli import run_pipeline
    from main.algorythms.area_selection.sar_gate_yolo.inference_gate import run_gate
except Exception:
    # Allow UI to start even if deps are missing; errors handled during run
    read_yaml = None
    run_pipeline = None
    run_gate = None


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
            self.progress.emit(15, "Подготовка конфига...")

            # Resolve config path inside local package
            import main.algorythms.area_selection.sar_gate_yolo as sar_pkg
            pkg_dir = os.path.dirname(sar_pkg.__file__)
            cfg_path = os.path.join(pkg_dir, 'config.yaml')
            if read_yaml is None or not os.path.exists(cfg_path):
                raise RuntimeError("Не найден sar_gate_yolo/config.yaml или зависимости не установлены")

            cfg = read_yaml(cfg_path)

            # Prepare temp dir for visualizations
            tmp_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'tmp')
            tmp_dir = os.path.abspath(tmp_dir)
            os.makedirs(tmp_dir, exist_ok=True)

            self.progress.emit(40, "Запуск алгоритма...")

            # Resolve YOLO model path
            yolo_rel = str(cfg.get('yolo', {}).get('onnx_path', 'models/yolo_cpu.onnx'))
            yolo_path = yolo_rel if os.path.isabs(yolo_rel) else os.path.join(pkg_dir, yolo_rel)
            use_yolo = os.path.exists(yolo_path) and run_pipeline is not None

            vis_out = os.path.join(tmp_dir, os.path.basename(self.image_path))

            if use_yolo:
                # Full pipeline: gate + crops + YOLO + NMS, with visualization
                results_map = run_pipeline(self.image_path, cfg_path, save_vis=tmp_dir, save_json_dir=None)
                dets = results_map.get(self.image_path, [])
                # Aggregate simple count (UI expects class/count table)
                objects = [{'class': 'Объекты', 'count': int(len(dets))}]
                results = {
                    'objects': objects,
                    'image_with_boxes': vis_out if os.path.exists(vis_out) else self.image_path
                }
            else:
                # Fallback: gate-only ROI proposal with rectangles
                self.progress.emit(60, "YOLO-модель не найдена, режим ROI...")
                img = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise RuntimeError("Не удалось загрузить изображение")
                gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                rois = run_gate(gray, cfg) if run_gate is not None else []
                vis = img.copy() if img.ndim == 3 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                for x1, y1, x2, y2 in [(r[0], r[1], r[2], r[3]) for r in rois]:
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.imwrite(vis_out, vis)
                results = {
                    'objects': [{'class': 'Зоны интереса', 'count': int(len(rois))}],
                    'image_with_boxes': vis_out
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
    
    def __init__(self, image_path, roi_type, sensitivity, bright_min, bright_max):
        super().__init__()
        self.image_path = image_path
        self.roi_type = roi_type
        self.sensitivity = sensitivity
        self.bright_min = int(max(0, min(255, bright_min)))
        self.bright_max = int(max(0, min(255, bright_max)))
        
    def run(self):
        try:
            self.progress.emit(15, "Подготовка конфига...")

            # Пути и конфиг
            import main.algorythms.area_selection.sar_gate_yolo as sar_pkg
            pkg_dir = os.path.dirname(sar_pkg.__file__)
            cfg_path = os.path.join(pkg_dir, 'config.yaml')
            if read_yaml is None or not os.path.exists(cfg_path):
                raise RuntimeError("Не найден sar_gate_yolo/config.yaml или зависимости не установлены")

            cfg = read_yaml(cfg_path)

            # Маппинг чувствительности (1..100) → плавные параметры CFAR/ROI/YOLO
            s = max(0.0, min(1.0, float(self.sensitivity) / 100.0))
            t = s ** 1.6  # сглаженная кривая

            # CFAR пороги: чем выше чувствительность, тем ниже пороги
            k_val = 5.0 - 3.0 * t            # 5.0 → 2.0
            cfar = cfg.setdefault('gate', {}).setdefault('cfar', {})
            cfar['k'] = float(max(1.0, min(8.0, k_val)))
            # Важно: оставляем q=0.5 (медиана) для быстрой реализации OS-CFAR
            cfar['q'] = float(cfg['gate']['cfar'].get('q', 0.5))

            # ROI параметры
            roi = cfg['gate'].setdefault('roi', {})
            base_min_area = int(roi.get('min_area', 12))
            roi['min_area'] = max(4, int(round(base_min_area * (1.0 - 0.6 * t))))  # 1.0x → 0.4x
            roi['dilate'] = int(max(1, min(3, round(1 + 2 * t))))                  # 1 → 3
            # Ограничим число ROI для ускорения (умеренно растёт с чувствительностью)
            base_topk = int(roi.get('topk', 150))
            roi['topk'] = max(40, min(base_topk, int(round(70 + 60 * t))))

            # YOLO параметры
            yolo = cfg.setdefault('yolo', {})
            yolo['conf_thresh'] = round(0.60 - 0.40 * t, 3)  # 0.60 → 0.20
            yolo['iou_nms'] = round(0.60 - 0.10 * t, 2)      # 0.60 → 0.50
            if t >= 0.6:
                yolo['use_soft_nms'] = True

            # Временный конфиг
            tmp_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'tmp')
            tmp_dir = os.path.abspath(tmp_dir)
            os.makedirs(tmp_dir, exist_ok=True)
            tmp_cfg_path = os.path.join(tmp_dir, 'config_runtime.yaml')
            with open(tmp_cfg_path, 'w') as f:
                yaml.safe_dump(cfg, f, allow_unicode=True)

            self.progress.emit(40, "Запуск детекции во вкладке ROI...")

            vis_out_dir = tmp_dir
            # Для визуализаций используем отдельный файл, чтобы не перетирать
            # исходное (например, уже улучшенное) изображение.
            base_name = os.path.basename(self.image_path)
            root, ext = os.path.splitext(base_name)
            vis_out = os.path.join(vis_out_dir, f"{root}_roi{ext or '.jpg'}")

            # ROI анализ с фильтром по яркости и подавлением пересечений
            img = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise RuntimeError("Не удалось загрузить изображение")
            gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rois_raw = run_gate(gray, cfg) if run_gate is not None else []

            # Простая NMS для ROI по IoU
            def iou(a, b):
                ax1, ay1, ax2, ay2 = a[0], a[1], a[2], a[3]
                bx1, by1, bx2, by2 = b[0], b[1], b[2], b[3]
                inter_x1 = max(ax1, bx1)
                inter_y1 = max(ay1, by1)
                inter_x2 = min(ax2, bx2)
                inter_y2 = min(ay2, by2)
                iw = max(0, inter_x2 - inter_x1)
                ih = max(0, inter_y2 - inter_y1)
                inter = iw * ih
                area_a = max(0, (ax2 - ax1)) * max(0, (ay2 - ay1))
                area_b = max(0, (bx2 - bx1)) * max(0, (by2 - by1))
                union = area_a + area_b - inter + 1e-6
                return inter / union

            rois_nms = []
            rois_sorted = sorted(rois_raw, key=lambda r: (r[2]-r[0])*(r[3]-r[1]), reverse=True)
            for r in rois_sorted:
                if all(iou(r, k) < 0.7 for k in rois_nms):
                    rois_nms.append(r)

            # Подсчет яркости каждой области и фильтрация по диапазону
            regions = []
            for (x1, y1, x2, y2) in [(r[0], r[1], r[2], r[3]) for r in rois_nms]:
                x1c, y1c = max(0, x1), max(0, y1)
                x2c, y2c = min(gray.shape[1], x2), min(gray.shape[0], y2)
                if x2c <= x1c or y2c <= y1c:
                    continue
                patch = gray[y1c:y2c, x1c:x2c]
                mean_brightness = float(np.mean(patch))
                if self.bright_min <= mean_brightness <= self.bright_max:
                    regions.append((x1c, y1c, x2c, y2c, mean_brightness))

            # Визуализация отфильтрованных областей и подписи яркости
            vis = img.copy() if img.ndim == 3 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            for x1, y1, x2, y2, mb in regions:
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(vis, f"{mb:.1f}", (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1, cv2.LINE_AA)
            cv2.imwrite(vis_out, vis)

            results = {
                'count': int(len(regions)),
                'image_with_boxes': vis_out,
                'regions': [{'x1':x1,'y1':y1,'x2':x2,'y2':y2,'brightness':mb} for x1,y1,x2,y2,mb in regions]
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
            
            # Импортируем алгоритм улучшения
            from main.algorythms.improvment.image_enhancement import ImageEnhancement
            
            # Создаем экземпляр алгоритма
            enhancer = ImageEnhancement()
            
            self.progress.emit(50, "Применение улучшений...")
            
            method_map = {
                # гибридное подавление шума (bilateral + NLM)
                'Гибридное подавление шума SAR': 'hybrid_sar_denoise',
                # адаптивный фильтр спекла
                'Адаптивное подавление спекла SAR': 'sar_adaptive',
                # анизотропная диффузия
                'Анизотропная диффузия SAR': 'sar_srad',
            }
            
            method = method_map.get(self.enhance_type, 'hybrid_sar_denoise')
            
            # Применяем улучшение
            enhanced_img, metrics = enhancer.enhance_image(
                self.image_path, 
                method=method, 
                intensity=self.intensity
            )
            
            if enhanced_img is None:
                raise RuntimeError("Не удалось улучшить изображение")
            
            # Сохраняем улучшенное изображение рядом с исходным (папка enhanced)
            output_dir = os.path.join(os.path.dirname(self.image_path), 'enhanced')
            os.makedirs(output_dir, exist_ok=True)
            
            base_name = os.path.splitext(os.path.basename(self.image_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}_enhanced.jpg")
            
            success = enhancer.save_enhanced_image(enhanced_img, output_path)
            if not success:
                raise RuntimeError("Не удалось сохранить улучшенное изображение")

            # Дополнительно сохраняем такое же улучшенное изображение в общую tmp‑директорию,
            # чтобы все промежуточные результаты были в одном месте.
            tmp_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'tmp')
            tmp_dir = os.path.abspath(tmp_dir)
            os.makedirs(tmp_dir, exist_ok=True)
            tmp_path = os.path.join(tmp_dir, os.path.basename(output_path))
            enhancer.save_enhanced_image(enhanced_img, tmp_path)
            
            self.progress.emit(100, "Улучшение завершено")
            
            # Формируем результаты
            results = {
                'enhanced_image': output_path,
                'quality_metrics': metrics or {
                    'psnr': 0,
                    'contrast_improvement': 0,
                    'brightness_change': 0
                }
            }
            
            self.finished.emit(results)
            
        except Exception as e:
            self.error.emit(str(e))


class YoloRoiInferenceWorker(QThread):
    """Инференс YOLOv11m по каждому ROI-кропу.

    На вход подаются кропы по bbox из анализа зон (кластеризация / GATE). Отдельный шаг
    улучшения качества на кропе перед инференсом не выполняется — только пиксели выреза
    области с того же снимка ``image_path``, на котором считался ROI.
    """
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(
        self,
        image_path: str,
        regions: list,
        weights_path: str,
        max_predictions: int = 3,
        confidence: float = 0.25,
        image_size: int = 1024,
    ):
        super().__init__()
        self.image_path = image_path
        self.regions = regions  # список словарей с ключами x1,y1,x2,y2,...
        self.weights_path = weights_path
        self.max_predictions = max_predictions
        self.confidence = confidence
        self.image_size = image_size

    def run(self):
        try:
            from pathlib import Path
            from PIL import Image
            from main.algorythms.area_selection.yolov11_inference import (
                detect_pil_image,
                select_torch_device,
            )

            self.progress.emit(15, "Загрузка YOLOv11m...")

            weights_p = Path(self.weights_path)
            if not weights_p.is_file():
                raise FileNotFoundError(f"Файл весов YOLOv11m не найден: {weights_p}")

            img_path = Path(self.image_path)
            if not img_path.is_file():
                raise FileNotFoundError(f"Изображение для YOLOv11m не найдено: {img_path}")

            full_img = Image.open(img_path)
            device = select_torch_device()

            roi_results = []
            total = max(1, len(self.regions))
            for i, r in enumerate(self.regions):
                frac = i / total
                self.progress.emit(20 + int(70 * frac), f"YOLO-инференс зоны {i+1}/{len(self.regions)}...")

                x1, y1, x2, y2 = int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"])
                w, h = full_img.size
                x1c, y1c = max(0, x1), max(0, y1)
                x2c, y2c = min(w, x2), min(h, y2)
                if x2c <= x1c or y2c <= y1c:
                    continue

                crop = full_img.crop((x1c, y1c, x2c, y2c))

                detect_res = detect_pil_image(
                    img=crop,
                    weights_path=weights_p,
                    conf=self.confidence,
                    imgsz=self.image_size,
                    max_predictions=self.max_predictions,
                    device=device,
                )
                preds = detect_res.get("predictions", [])
                best = detect_res.get("best")
                roi_results.append(
                    {
                        "box": (x1c, y1c, x2c, y2c),
                        "predictions": preds,
                        "best": best,
                    }
                )

            self.progress.emit(100, "YOLO-инференс зон завершён")
            self.finished.emit(
                {
                    "image": str(img_path),
                    "weights": str(weights_p),
                    "device": device,
                    "roi_results": roi_results,
                }
            )

        except Exception as e:
            self.error.emit(str(e))


class NewController:
    def __init__(self, view):
        self.view = view
        self.current_workers = []
        # Флаг и путь для последовательного пайплайна улучшение → ROI → YOLO
        self.pipeline_active = False
        self.pipeline_image_path = None
        # Последние результаты анализа ROI (для YOLOv11m-инференса)
        self.last_roi_image_path: Optional[str] = None
        self.last_roi_regions: list = []
        # Результаты YOLOv11m по зонам и текущий индекс
        self.sarhub_roi_results: list = []
        self.sarhub_roi_index: int = 0
        self.setup_connections()

    # ---------- Вспомогательные безопасные методы для UI ----------

    def _safe_set_enabled(self, widget, enabled: bool):
        """Безопасно включает/выключает виджеты, игнорируя случаи,
        когда C++‑объект уже удалён (RuntimeError: wrapped C/C++ object ...)."""
        if widget is None:
            return
        try:
            widget.setEnabled(enabled)
        except RuntimeError:
            # Виджет уже уничтожен Qt (например, окно закрыто) — просто игнорируем
            pass
        
    def setup_connections(self):
        """Настройка соединений между интерфейсом и контроллером"""
        
        # Главная вкладка - загрузка/сохранение; детекция доступна во вкладке ROI
        self.view.browse_btn.clicked.connect(self.browse_files_main)
        self.view.detect_btn.setEnabled(False)
        self.view.detect_btn.setToolTip("Детекция доступна во вкладке 'Зоны интересов'")
        self.view.save_results_btn.clicked.connect(self.save_detection_results)

        # YOLOv11m-инференс по ROI на главной вкладке
        if hasattr(self.view, "sarhub_classify_btn"):
            self.view.sarhub_classify_btn.clicked.connect(self.start_sarhub_classification)
        # Навигация по зонам YOLOv11m
        if hasattr(self.view, "sarhub_prev_btn"):
            self.view.sarhub_prev_btn.clicked.connect(self.show_prev_sarhub_roi)
        if hasattr(self.view, "sarhub_next_btn"):
            self.view.sarhub_next_btn.clicked.connect(self.show_next_sarhub_roi)
        # Улучшение текущего фрагмента (ROI) в окне "Предсказание модели"
        if hasattr(self.view, "roi_frame_enhance_btn"):
            self.view.roi_frame_enhance_btn.clicked.connect(self.enhance_current_roi_frame_visual)
        
        # Вкладка зон интересов
        self.view.roi_analyze_btn.clicked.connect(self.start_roi_analysis)
        
        # Вкладка улучшения качества
        self.view.enhance_btn.clicked.connect(self.start_enhancement)
        # Полный конвейер: улучшение → ROI → YOLO
        if hasattr(self.view, "enhance_pipeline_btn"):
            self.view.enhance_pipeline_btn.clicked.connect(self.start_full_pipeline)
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
                
    def _run_detection_for_image(self, image_path: str):
        """Внутренний запуск детекции объектов для указанного изображения"""
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
        
    def start_detection(self):
        """Запуск детекции объектов для первого изображения в списке"""
        if self.view.file_list.count() == 0:
            self.view.show_error("Выберите изображения для анализа")
            return
            
        item = self.view.file_list.item(0)
        if not item:
            return
        image_path = item.data(1)
        self._run_detection_for_image(image_path)

    def start_sarhub_classification(self):
        """Запуск YOLOv11m по найденным зонам интересов (ROI)
        
        Используются результаты последнего анализа во вкладке 'Зоны интересов'.
        """
        if not self.last_roi_image_path or not self.last_roi_regions:
            self.view.show_error(
                "Сначала выполните анализ зон интересов во вкладке 'Зоны интересов',\n"
                "затем запустите YOLOv11m-инференс."
            )
            return

        image_path = self.last_roi_image_path

        # Путь к весам YOLOv11m: Веса/YOLOv11m/best.pt
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        default_weights = os.path.join(root_dir, "Веса", "YOLOv11m", "best.pt")

        weights_path = default_weights

        if not os.path.isfile(weights_path):
            self.view.show_error(
                f"Не найден файл весов YOLOv11m.\n"
                f"Ожидаемый путь: {weights_path}\n"
                f"Проверьте, что файл best.pt лежит в папке Веса/YOLOv11m."
            )
            return

        # Сбрасываем предыдущие результаты классификации
        self.sarhub_roi_results = []
        self.sarhub_roi_index = 0

        self.set_ui_enabled(False)
        self.view.sarhub_status_label.setText("YOLOv11m-инференс зон запущен...")

        worker = YoloRoiInferenceWorker(
            image_path=image_path,
            regions=self.last_roi_regions,
            weights_path=weights_path,
            max_predictions=3,
            confidence=0.25,
            image_size=1024,
        )
        worker.progress.connect(self.view.update_progress)
        worker.finished.connect(self.on_sarhub_finished)
        worker.error.connect(self.on_sarhub_error)

        self.current_workers.append(worker)
        worker.start()

    def on_sarhub_finished(self, result: dict):
        """Обработка завершения YOLOv11m по зонам ROI
        
        Формируем отдельные изображения‑кропы для каждой зоны с подписью класса
        и даём возможность листать их стрелками в окне 'Предсказание модели'.
        """
        self.set_ui_enabled(True)
        self.view.progress_bar.setVisible(False)

        image_path = result.get("image")
        roi_results = result.get("roi_results", [])

        if not image_path or not os.path.isfile(image_path) or not roi_results:
            self.view.sarhub_status_label.setText("YOLOv11m завершён (зон нет)")
            # Сбрасываем отображение
            self.view.sarhub_roi_info_label.setText("Зона 0 / 0")
            return

        # Готовим директорию и исходное изображение для кропов
        tmp_dir = os.path.join(os.path.dirname(__file__), "..", "..", "tmp")
        tmp_dir = os.path.abspath(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            self.view.sarhub_status_label.setText("YOLOv11m завершён (ошибка чтения изображения)")
            self.view.sarhub_roi_info_label.setText("Зона 0 / 0")
            return

        # Приводим изображение к BGR один раз (без рамок и дополнительных аннотаций)
        base_vis = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # Копия для полноразмерной визуализации с боксами и подписями классов
        full_vis = base_vis.copy()

        prepared = []
        annotated_full_path = None
        for idx, roi in enumerate(roi_results):
            box = roi.get("box")
            best = roi.get("best")
            if not box:
                continue
            x1, y1, x2, y2 = map(int, box)
            h, w = base_vis.shape[:2]
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(w, x2), min(h, y2)
            if x2c <= x1c or y2c <= y1c:
                continue

            # Кроп тех же координат, что после кластеризации/ROI на ``image_path``, без фильтров
            crop = base_vis[y1c:y2c, x1c:x2c].copy()

            # Подпись основного класса поверх блока (top‑1 для наглядности)
            label = best.get("label") if best else "неизвестно"
            prob = best.get("prob") if best and "prob" in best else None
            if label is None:
                label = "неизвестно"
            text = label if prob is None else f"{label} ({prob:.2f})"
            inner_bbox = best.get("bbox") if best else None
            if inner_bbox:
                bx1, by1, bx2, by2 = map(int, inner_bbox)
                ch, cw = crop.shape[:2]
                bx1, by1 = max(0, bx1), max(0, by1)
                bx2, by2 = min(cw, bx2), min(ch, by2)
                if bx2 > bx1 and by2 > by1:
                    cv2.rectangle(crop, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                    cv2.rectangle(
                        full_vis,
                        (x1c + bx1, y1c + by1),
                        (x1c + bx2, y1c + by2),
                        (0, 255, 0),
                        2,
                    )

            y_text = max(16, int(0.08 * (y2c - y1c)))
            # Тень
            cv2.putText(
                crop,
                text,
                (5, y_text + 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            # Основной текст
            cv2.putText(
                crop,
                text,
                (5, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Рисуем bbox и подпись для этой зоны на полноразмерном изображении
            cv2.rectangle(full_vis, (x1c, y1c), (x2c, y2c), (0, 255, 255), 2)
            full_y_text = max(16, y1c - 4)
            cv2.putText(
                full_vis,
                text,
                (x1c + 1, full_y_text + 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                full_vis,
                text,
                (x1c, full_y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

            out_path = os.path.join(tmp_dir, f"sarhub_roi_{idx+1}.jpg")
            cv2.imwrite(out_path, crop)

            prepared.append(
                {
                    "box": (x1c, y1c, x2c, y2c),
                    "best": best,
                    "predictions": roi.get("predictions", []),
                    "crop_path": out_path,
                }
            )

        # Сохраняем полноразмерное изображение с выделенными классами
        if len(roi_results) > 0:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            annotated_full_path = os.path.join(tmp_dir, f"{base_name}_sarhub_full.jpg")
            cv2.imwrite(annotated_full_path, full_vis)

        self.sarhub_roi_results = prepared
        self.sarhub_roi_index = 0

        total_rois = len(self.sarhub_roi_results)
        if total_rois == 0:
            self.view.sarhub_status_label.setText("YOLOv11m завершён (кропы не сформированы)")
            self.view.sarhub_roi_info_label.setText("Зона 0 / 0")
        else:
            # Обновляем окно "Найденные объекты": зона + до трёх лучших YOLO-детекций.
            table = self.view.results_table
            
            max_preds = 3  # всегда показываем top‑3
            col_count = 1 + max_preds  # Зона + 3 класса
            table.setColumnCount(col_count)

            # Заголовки: "Зона", "Класс 1", "Класс 2", ...
            headers = ["Зона"] + [f"Объект {i+1}" for i in range(max_preds)]
            table.setHorizontalHeaderLabels(headers)

            table.setRowCount(total_rois)
            for i, roi in enumerate(self.sarhub_roi_results):
                table.setItem(i, 0, QTableWidgetItem(f"Зона {i+1}"))

                preds = roi.get("predictions", [])
                for j in range(max_preds):
                    if j < len(preds):
                        p = preds[j]
                        lbl = p.get("label", "неизвестно")
                        pr = p.get("prob")
                        if pr is not None and pr > 0.001:  # Если вероятность > 0.001, показываем класс
                            cell_text = f"{lbl} ({pr:.2f})"
                        else:
                            cell_text = "0.00"
                    else:
                        # Если предсказаний меньше трёх — заполняем "нулями"
                        cell_text = "0.00"
                    table.setItem(i, 1 + j, QTableWidgetItem(cell_text))
            
            # Настройка ширины столбцов: автоматическое подстраивание под содержимое
            from PyQt5.QtWidgets import QHeaderView
            for col in range(col_count):
                table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeToContents)
                # Минимальная ширина для каждого столбца
                if col == 0:
                    table.setColumnWidth(col, max(80, table.columnWidth(col)))
                else:
                    table.setColumnWidth(col, max(120, table.columnWidth(col)))

            # Показать первую зону в окне предсказания
            self._show_current_sarhub_roi()
            device = result.get("device", "cpu")
            self.view.sarhub_status_label.setText(f"YOLOv11m-инференс зон завершён (device: {device})")

            # Показать полноразмерное изображение с выделенными классами
            if annotated_full_path is not None:
                try:
                    self.view.original_view.set_image(annotated_full_path)
                except RuntimeError:
                    # Окно могло быть уже закрыто, просто игнорируем
                    pass

        # Если это был шаг полного пайплайна — завершаем его
        if self.pipeline_active:
            self.pipeline_active = False
            self.pipeline_image_path = None

    def on_sarhub_error(self, error_message: str):
        """Обработка ошибок YOLOv11m-инференса"""
        self.set_ui_enabled(True)
        self.view.progress_bar.setVisible(False)
        self.view.sarhub_status_label.setText("Ошибка YOLOv11m-инференса")
        self.view.show_error(f"Ошибка YOLOv11m-инференса: {error_message}")
        
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
        
        # Завершаем последовательный пайплайн, если он был активен
        if self.pipeline_active:
            self.pipeline_active = False
            self.pipeline_image_path = None
            
    def on_detection_error(self, error_message):
        """Обработка ошибки детекции"""
        self.set_ui_enabled(True)
        self.view.progress_bar.setVisible(False)
        # Добавляем подсказку по установке моделей и зависимостей
        hint = "\n\nПодсказка: убедитесь, что установлен onnxruntime (pip install -r requirements.txt) и файл модели YOLO расположен по пути main/algorythms/area_selection/sar_gate_yolo/models/yolo_cpu.onnx."
        self.view.show_error(f"Ошибка детекции: {error_message}{hint}")
        
    def _run_roi_analysis_for_image(
        self,
        image_path: str,
        *,
        roi_type: Optional[str] = None,
        sensitivity: Optional[int] = None,
        bright_min: Optional[int] = None,
        bright_max: Optional[int] = None,
    ):
        """Запуск анализа зон интересов. Параметры ROI можно передать явно (например, снимок
        настроек на момент нажатия «Полный SAR пайплайн» до отключения виджетов)."""
        # Всегда читаем настройки до set_ui_enabled(False): у отключённых слайдеров
        # значение обычно корректное, но явный снимок исключает артефакты UI.
        eff_roi_type = roi_type if roi_type is not None else self.view.roi_type_combo.currentText()
        eff_sensitivity = (
            sensitivity if sensitivity is not None else self.view.roi_sensitivity_slider.value()
        )
        eff_bright_min = (
            bright_min if bright_min is not None else self.view.roi_brightness_min_slider.value()
        )
        eff_bright_max = (
            bright_max if bright_max is not None else self.view.roi_brightness_max_slider.value()
        )

        self.last_roi_image_path = image_path

        self.set_ui_enabled(False)

        worker = ROIWorker(
            image_path,
            eff_roi_type,
            eff_sensitivity,
            eff_bright_min,
            eff_bright_max,
        )
        worker.progress.connect(self.view.update_progress)
        worker.finished.connect(self.on_roi_finished)
        worker.error.connect(self.on_roi_error)
        
        self.current_workers.append(worker)
        worker.start()
        
        # Показываем исходное изображение
        self.view.roi_source_view.set_image(image_path)
        
    def start_roi_analysis(self):
        """Запуск анализа зон интересов для первого изображения в списке"""
        if self.view.file_list.count() == 0:
            self.view.show_error("Выберите изображения для анализа")
            return
            
        item = self.view.file_list.item(0)
        if not item:
            return
        image_path = item.data(1)
        self._run_roi_analysis_for_image(image_path)
        
    def on_roi_finished(self, results):
        """Обработка завершения анализа ROI"""
        self.set_ui_enabled(True)
        self.view.progress_bar.setVisible(False)
        
        # Показываем детекцию во вкладке ROI
        self.view.roi_results_list.clear()
        # Сохраняем регионы для последующего YOLOv11m-инференса
        self.last_roi_regions = results.get("regions", [])
        if 'image_with_boxes' in results:
            cnt = int(results.get('count', 0))
            self.view.roi_results_list.addItem(f"Найдено областей: {cnt}")
            # Выводим яркость каждой области
            if 'regions' in results:
                for r in results['regions']:
                    self.view.roi_results_list.addItem(f"({r['x1']},{r['y1']})-({r['x2']},{r['y2']}): ярк. {r['brightness']:.1f}")
            self.view.roi_analysis_view.set_image(results['image_with_boxes'])
        elif 'roi_image' in results:
            # Fallback старого формата
            self.view.roi_analysis_view.set_image(results['roi_image'])
        
        # Если активен полнофункциональный пайплайн,
        # после ROI сразу запускаем YOLOv11m по найденным областям.
        if self.pipeline_active and self.pipeline_image_path:
            # Убедимся, что путь и регионы заданы
            self.last_roi_image_path = self.pipeline_image_path
            self.last_roi_regions = results.get("regions", [])
            if self.last_roi_image_path and self.last_roi_regions:
                self.start_sarhub_classification()
            
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
        
    def start_full_pipeline(self):
        """
        Полный SAR‑пайплайн:
        1) Улучшение изображения выбранным алгоритмом;
        2) Анализ зон интересов (ROI) по улучшенному снимку;
        3) YOLOv11m-инференс каждой найденной зоны
           и просмотр результатов во вкладке «Предсказание модели» (стрелки ← / →).
        """
        if self.view.file_list.count() == 0:
            self.view.show_error("Выберите изображения для анализа")
            return
            
        item = self.view.file_list.item(0)
        if not item:
            return
            
        image_path = item.data(1)
        enhance_type = self.view.enhance_type_combo.currentText()
        intensity = self.view.enhance_intensity_slider.value()

        # Снимок настроек ROI до любого set_ui_enabled(False) — они же,
        # что сейчас на вкладке «Зоны интересов».
        roi_type_snap = self.view.roi_type_combo.currentText()
        sensitivity_snap = self.view.roi_sensitivity_slider.value()
        bright_min_snap = self.view.roi_brightness_min_slider.value()
        bright_max_snap = self.view.roi_brightness_max_slider.value()

        try:
            # Импортируем алгоритм улучшения
            from main.algorythms.improvment.image_enhancement import ImageEnhancement
            enhancer = ImageEnhancement()
            
            # Маппинг текста интерфейса на внутренние методы
            method_map = {
                'Гибридное подавление шума SAR': 'hybrid_sar_denoise',
                'Адаптивное подавление спекла SAR': 'sar_adaptive',
                'Анизотропная диффузия SAR': 'sar_srad',
            }
            method = method_map.get(enhance_type, 'hybrid_sar_denoise')
            
            # Шаг 1: улучшение изображения (синхронно в GUI‑потоке, для простоты)
            self.set_ui_enabled(False)
            enhanced_img, metrics = enhancer.enhance_image(
                image_path,
                method=method,
                intensity=intensity,
            )
            if enhanced_img is None:
                self.set_ui_enabled(True)
                self.view.show_error("Не удалось улучшить изображение в полном пайплайне")
                return
            
            # Сохраняем улучшенное изображение во временную директорию
            tmp_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'tmp')
            tmp_dir = os.path.abspath(tmp_dir)
            os.makedirs(tmp_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            enhanced_path = os.path.join(tmp_dir, f"{base_name}_pipeline_enhanced.jpg")
            enhancer.save_enhanced_image(enhanced_img, enhanced_path)
            
            # Обновляем вкладку улучшения
            self.view.enhance_original_view.set_image(image_path)
            self.view.enhance_result_view.set_image(enhanced_path)
            if metrics:
                quality_text = f"Метрики качества SAR улучшения (полный пайплайн):\n\n"
                quality_text += f"PSNR: {metrics.get('psnr', 0):.2f} dB\n"
                quality_text += f"Улучшение контраста: {metrics.get('contrast_improvement', 0):.1f}%\n"
                quality_text += f"Изменение яркости: {metrics.get('brightness_change', 0):.1f}%\n"
                quality_text += f"Оригинальный контраст: {metrics.get('original_contrast', 0):.2f}\n"
                quality_text += f"Улучшенный контраст: {metrics.get('enhanced_contrast', 0):.2f}"
                self.view.quality_info.setText(quality_text)
            
            # Включаем режим последовательного пайплайна и запоминаем путь
            self.pipeline_active = True
            self.pipeline_image_path = enhanced_path

            # Шаг 2: ROI‑анализ по улучшенному изображению (настройки ROI — с момента клика)
            self._run_roi_analysis_for_image(
                enhanced_path,
                roi_type=roi_type_snap,
                sensitivity=sensitivity_snap,
                bright_min=bright_min_snap,
                bright_max=bright_max_snap,
            )
            
        except Exception as e:
            self.set_ui_enabled(True)
            self.view.show_error(f"Ошибка полного пайплайна: {e}")
        
    def on_enhancement_finished(self, results):
        """Обработка завершения улучшения"""
        self.set_ui_enabled(True)
        self.view.progress_bar.setVisible(False)
        
        # Показываем улучшенное изображение
        if 'enhanced_image' in results:
            self.view.enhance_result_view.set_image(results['enhanced_image'])
            
        # Показываем метрики качества в интерфейсе
        if 'quality_metrics' in results:
            metrics = results['quality_metrics']
            quality_text = f"Метрики качества SAR улучшения:\n\n"
            quality_text += f"PSNR: {metrics.get('psnr', 0):.2f} dB\n"
            quality_text += f"Улучшение контраста: {metrics.get('contrast_improvement', 0):.1f}%\n"
            quality_text += f"Изменение яркости: {metrics.get('brightness_change', 0):.1f}%\n"
            quality_text += f"Оригинальный контраст: {metrics.get('original_contrast', 0):.2f}\n"
            quality_text += f"Улучшенный контраст: {metrics.get('enhanced_contrast', 0):.2f}"
            
            # Обновляем информацию о качестве в интерфейсе
            self.view.quality_info.setText(quality_text)
            
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
            
    def enhance_current_roi_frame_visual(self):
        """
        Улучшение качества текущего маленького фрагмента (ROI), который показан
        в окне 'Предсказание модели' на главной вкладке.
        Влияет только на визуализацию, не на YOLOv11m-инференс.
        """
        if not self.sarhub_roi_results:
            self.view.show_error("Нет фрагментов для улучшения. Сначала выполните полный SAR‑пайплайн.")
            return

        idx = max(0, min(self.sarhub_roi_index, len(self.sarhub_roi_results) - 1))
        roi = self.sarhub_roi_results[idx]
        crop_path = roi.get("crop_path")
        if not crop_path or not os.path.isfile(crop_path):
            self.view.show_error("Не удалось найти изображение текущего фрагмента.")
            return

        img = cv2.imread(crop_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            self.view.show_error("Ошибка загрузки изображения фрагмента.")
            return

        # Настройки метода/интенсивности из нового блока на главной вкладке
        method_text = getattr(self.view, "roi_frame_method_combo", None)
        if method_text is not None:
            method_text = self.view.roi_frame_method_combo.currentText()
        else:
            method_text = "Гибридное подавление шума SAR"

        intensity_slider = getattr(self.view, "roi_frame_intensity_slider", None)
        intensity = intensity_slider.value() if intensity_slider is not None else 50

        method_map = {
            'Гибридное подавление шума SAR': 'hybrid_sar_denoise',
            'Адаптивное подавление спекла SAR': 'sar_adaptive',
            'Анизотропная диффузия SAR': 'sar_srad',
        }
        method = method_map.get(method_text, 'hybrid_sar_denoise')

        try:
            from main.algorythms.improvment.image_enhancement import ImageEnhancement
            enhancer = ImageEnhancement()
            enhanced_np = enhancer.enhance_array(img, method=method, intensity=intensity)

            tmp_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'tmp')
            tmp_dir = os.path.abspath(tmp_dir)
            os.makedirs(tmp_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(crop_path))[0]
            out_path = os.path.join(tmp_dir, f"{base_name}_vis_enhanced.jpg")
            cv2.imwrite(out_path, enhanced_np)

            # Показываем улучшенный фрагмент в окне предсказания
            self.view.prediction_view.set_image(out_path)
        except Exception as e:
            self.view.show_error(f"Не удалось улучшить фрагмент: {e}")

    def set_ui_enabled(self, enabled):
        """Включение/отключение элементов интерфейса"""
        # Главная вкладка
        self.view.detect_btn.setEnabled(enabled)
        self.view.browse_btn.setEnabled(enabled)
        self.view.save_results_btn.setEnabled(enabled)
        self.view.confidence_slider.setEnabled(enabled)
        if hasattr(self.view, "roi_frame_method_combo"):
            self.view.roi_frame_method_combo.setEnabled(enabled)
        if hasattr(self.view, "roi_frame_intensity_slider"):
            self.view.roi_frame_intensity_slider.setEnabled(enabled)
        if hasattr(self.view, "roi_frame_enhance_btn"):
            self.view.roi_frame_enhance_btn.setEnabled(enabled)
        
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
        if hasattr(self.view, "enhance_pipeline_btn"):
            self.view.enhance_pipeline_btn.setEnabled(enabled)
        # YOLOv11m-инференс
        if hasattr(self.view, "sarhub_classify_btn"):
            self.view.sarhub_classify_btn.setEnabled(enabled)
        if hasattr(self.view, "sarhub_prev_btn"):
            self.view.sarhub_prev_btn.setEnabled(enabled)
        if hasattr(self.view, "sarhub_next_btn"):
            self.view.sarhub_next_btn.setEnabled(enabled)
        
        if not enabled:
            self.view.progress_bar.setVisible(True)
        else:
            self.view.progress_bar.setVisible(False)

    # ---------- Навигация по результатам YOLOv11m ----------

    def _show_current_sarhub_roi(self):
        """Показывает текущий ROI‑кроп и обновляет подписи"""
        total = len(self.sarhub_roi_results)
        if total == 0:
            self.view.sarhub_roi_info_label.setText("Зона 0 / 0")
            return

        idx = max(0, min(self.sarhub_roi_index, total - 1))
        self.sarhub_roi_index = idx
        roi = self.sarhub_roi_results[idx]

        crop_path = roi.get("crop_path")
        if crop_path and os.path.isfile(crop_path):
            self.view.prediction_view.set_image(crop_path)

        self.view.sarhub_roi_info_label.setText(f"Зона {idx+1} / {total}")

        # Формируем текст: топ-N объектов для текущего блока
        preds = roi.get("predictions", [])
        if preds:
            parts = []
            for p in preds:
                lbl = p.get("label", "неизвестно")
                pr = p.get("prob")
                if pr is not None:
                    parts.append(f"{lbl} ({pr:.2f})")
                else:
                    parts.append(lbl)
            summary = "; ".join(parts)
            self.view.sarhub_status_label.setText(f"Зона {idx+1}/{total}: {summary}")
        else:
            self.view.sarhub_status_label.setText(f"Зона {idx+1}/{total}: результаты отсутствуют")

    def show_next_sarhub_roi(self):
        """Показать следующую зону YOLOv11m"""
        if not self.sarhub_roi_results:
            return
        self.sarhub_roi_index = (self.sarhub_roi_index + 1) % len(self.sarhub_roi_results)
        self._show_current_sarhub_roi()

    def show_prev_sarhub_roi(self):
        """Показать предыдущую зону YOLOv11m"""
        if not self.sarhub_roi_results:
            return
        self.sarhub_roi_index = (self.sarhub_roi_index - 1) % len(self.sarhub_roi_results)
        self._show_current_sarhub_roi()


