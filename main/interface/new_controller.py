import os
import sys
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
            vis_out = os.path.join(vis_out_dir, os.path.basename(self.image_path))

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
            
            # Сохраняем улучшенное изображение
            output_dir = os.path.join(os.path.dirname(self.image_path), 'enhanced')
            os.makedirs(output_dir, exist_ok=True)
            
            base_name = os.path.splitext(os.path.basename(self.image_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}_enhanced.jpg")
            
            success = enhancer.save_enhanced_image(enhanced_img, output_path)
            
            if not success:
                raise RuntimeError("Не удалось сохранить улучшенное изображение")
            
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


class NewController:
    def __init__(self, view):
        self.view = view
        self.current_workers = []
        # Флаг и путь для последовательного пайплайна улучшение → ROI → YOLO
        self.pipeline_active = False
        self.pipeline_image_path = None
        self.setup_connections()
        
    def setup_connections(self):
        """Настройка соединений между интерфейсом и контроллером"""
        
        # Главная вкладка - загрузка/сохранение; детекция доступна во вкладке ROI
        self.view.browse_btn.clicked.connect(self.browse_files_main)
        self.view.detect_btn.setEnabled(False)
        self.view.detect_btn.setToolTip("Детекция доступна во вкладке 'Зоны интересов'")
        self.view.send_coords_btn.clicked.connect(self.send_coordinates)
        self.view.save_results_btn.clicked.connect(self.save_detection_results)
        
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
        
    def _run_roi_analysis_for_image(self, image_path: str):
        """Внутренний запуск анализа зон интересов для указанного изображения"""
        roi_type = self.view.roi_type_combo.currentText()
        sensitivity = self.view.roi_sensitivity_slider.value()
        
        self.set_ui_enabled(False)
        
        # Яркость из интерфейса
        bright_min = self.view.roi_brightness_min_slider.value()
        bright_max = self.view.roi_brightness_max_slider.value()
        worker = ROIWorker(image_path, roi_type, sensitivity, bright_min, bright_max)
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
        
        # Если активен полнофункциональный пайплайн, запускаем следующий этап — детекцию (YOLO)
        if self.pipeline_active and self.pipeline_image_path:
            self._run_detection_for_image(self.pipeline_image_path)
            
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
        2) Анализ зон интересов (ROI);
        3) Детекция / классификация (YOLO) по улучшенному изображению.
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
            
            # Шаг 2: ROI‑анализ по улучшенному изображению (асинхронно)
            self._run_roi_analysis_for_image(enhanced_path)
            
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
        if hasattr(self.view, "enhance_pipeline_btn"):
            self.view.enhance_pipeline_btn.setEnabled(enabled)
        
        if not enabled:
            self.view.progress_bar.setVisible(True)
        else:
            self.view.progress_bar.setVisible(False)


