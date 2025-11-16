"""
Алгоритм улучшения качества радиолокационных (SAR) снимков.
Перенесён в пакет main, чтобы не зависеть от других директорий.
"""

import cv2
import numpy as np
from PIL import Image
import os
import tempfile

from .denoising_models import DenoisingModelManager, create_fallback_denoising


class ImageEnhancement:
    """Класс для улучшения качества радиолокационных (SAR) снимков.

    Специализированные алгоритмы для обработки спутниковых радиолокационных данных:
    - Адаптивное осветление темных областей
    - Специализированное подавление шума для SAR
    - Улучшение контраста с учетом характеристик радиолокационных отражений
    - Автоматический выбор параметров на основе анализа изображения
    """

    def __init__(self):
        self.enhancement_methods = {
            "baseline": self.baseline_enhancement,
            "denoise": self.denoise_image,
            "sharpen": self.sharpen_image,
            "contrast": self.enhance_contrast,
            "sar_enhance": self.sar_enhancement,
            "sar_denoise": self.sar_denoise,
            "sar_brighten": self.sar_brighten,
            # Новые методы специально для SAR
            "sar_auto_enhance": self.sar_auto_enhance,
            "sar_contrast": self.sar_contrast,
            "sar_sharpen": self.sar_sharpen,
            "sar_comprehensive": self.sar_comprehensive,
            # Методы с нейронными сетями
            "sar_ai_denoise": self.sar_ai_denoise,
            "sar_ai_enhance": self.sar_ai_enhance,
            # Новый алгоритм улучшения SAR
            "sar_morphological_enhance": self.sar_morphological_enhance,
        }

        # Инициализация менеджера моделей
        self.model_manager = DenoisingModelManager()
        self.fallback_denoising = create_fallback_denoising()

        # Попытка загрузки предобученных моделей
        self._initialize_models()

        # Параметры для обработки больших изображений
        self.max_tile_size = 4096  # Максимальный размер тайла для обработки
        self.overlap_size = 128  # Размер перекрытия между тайлами
        self.max_memory_size = 8192  # Максимальный размер для обработки в памяти (в пикселях)

    def _process_with_size_support(self, image_path, process_func, intensity=50):
        """Универсальная обертка для обработки изображений любого размера"""
        try:
            # Загружаем изображение
            img, width, height, pil_img = self._load_image_any_size(image_path)

            # Если изображение слишком большое, используем tiled обработку
            if img is None and pil_img is not None:
                return self._process_large_image_tiled(image_path, process_func, intensity)

            # Обычная обработка в памяти
            if img is None:
                img = cv2.imread(image_path)
                if img is None:
                    raise ValueError(f"Не удалось загрузить изображение: {image_path}")

            # Применяем функцию обработки
            return process_func(img, intensity)

        except Exception as e:
            print(f"Ошибка в _process_with_size_support: {e}")
            return None

    def baseline_enhancement(self, image_path, intensity=50):
        """
        Базовый пайплайн улучшения:
        1. Подавление шума (Non-local Means)
        2. Увеличение резкости
        """

        def process_func(img, intensity):
            # 1. Подавление шума (Non-local Means Denoising)
            img_denoised = cv2.fastNlMeansDenoisingColored(
                img,
                None,
                h=10,
                hColor=10,
                templateWindowSize=7,
                searchWindowSize=21,
            )

            # 2. Увеличение резкости
            img_sharp = self._apply_sharpening(img_denoised, intensity)

            return img_sharp

        try:
            return self._process_with_size_support(image_path, process_func, intensity)
        except Exception as e:
            print(f"Ошибка в baseline_enhancement: {e}")
            return None

    def _load_image_any_size(self, image_path):
        """Загрузка изображения любого размера с поддержкой больших файлов"""
        try:
            # Используем PIL для загрузки больших изображений
            pil_img = Image.open(image_path)

            # Получаем размеры
            width, height = pil_img.size

            # Если изображение очень большое, загружаем как массив numpy напрямую
            if width * height > self.max_memory_size * self.max_memory_size:
                # Для очень больших изображений используем tiling
                return None, width, height, pil_img

            # Конвертируем в numpy array
            img_array = np.array(pil_img)

            # Конвертируем в BGR для OpenCV
            if len(img_array.shape) == 3:
                if img_array.shape[2] == 4:  # RGBA
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
                elif img_array.shape[2] == 3:  # RGB
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            return img_array, width, height, None

        except Exception:
            # Fallback на OpenCV
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Не удалось загрузить изображение: {image_path}")
            height, width = img.shape[:2]
            return img, width, height, None

    def _process_large_image_tiled(self, image_path, process_func, intensity=50):
        """Обработка большого изображения по частям (tiling)"""
        try:
            pil_img = Image.open(image_path)
            width, height = pil_img.size

            # Определяем размер тайлов
            tile_size = min(self.max_tile_size, width, height)
            overlap = self.overlap_size

            # Создаем выходное изображение
            if pil_img.mode == "RGB":
                result_img = Image.new("RGB", (width, height))
            elif pil_img.mode == "RGBA":
                result_img = Image.new("RGBA", (width, height))
            else:
                result_img = Image.new("L", (width, height))

            # Обрабатываем по тайлам
            num_tiles_x = (width + tile_size - overlap - 1) // (tile_size - overlap)
            num_tiles_y = (height + tile_size - overlap - 1) // (tile_size - overlap)
            for ty in range(num_tiles_y):
                for tx in range(num_tiles_x):
                    # Вычисляем координаты тайла с перекрытием
                    x1 = max(0, tx * (tile_size - overlap))
                    y1 = max(0, ty * (tile_size - overlap))
                    x2 = min(width, x1 + tile_size)
                    y2 = min(height, y1 + tile_size)

                    # Вырезаем тайл
                    tile = pil_img.crop((x1, y1, x2, y2))

                    # Сохраняем временный тайл
                    temp_dir = tempfile.gettempdir()
                    temp_path = os.path.join(
                        temp_dir, f"tile_{tx}_{ty}_{os.getpid()}.png"
                    )
                    tile.save(temp_path)

                    # Загружаем тайл и обрабатываем
                    tile_img = cv2.imread(temp_path)
                    if tile_img is not None:
                        processed_tile = process_func(tile_img, intensity)
                    else:
                        processed_tile = None

                    if processed_tile is not None:
                        # Конвертируем обратно в PIL Image
                        if len(processed_tile.shape) == 3:
                            processed_pil = Image.fromarray(
                                cv2.cvtColor(processed_tile, cv2.COLOR_BGR2RGB)
                            )
                        else:
                            processed_pil = Image.fromarray(processed_tile)

                        # Вычисляем область без перекрытия для вставки
                        insert_x1 = overlap // 2 if tx > 0 else 0
                        insert_y1 = overlap // 2 if ty > 0 else 0
                        insert_x2 = processed_pil.width - (
                            overlap // 2 if tx < num_tiles_x - 1 else 0
                        )
                        insert_y2 = processed_pil.height - (
                            overlap // 2 if ty < num_tiles_y - 1 else 0
                        )

                        # Вставляем обработанный тайл
                        result_img.paste(
                            processed_pil.crop(
                                (insert_x1, insert_y1, insert_x2, insert_y2)
                            ),
                            (x1 + insert_x1, y1 + insert_y1),
                        )

                    # Удаляем временный файл
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass

            # Конвертируем результат в numpy array для совместимости
            result_array = np.array(result_img)
            if len(result_array.shape) == 3:
                result_array = cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)

            return result_array

        except Exception as e:
            print(f"Ошибка в tiled обработке: {e}")
            return None

    def _apply_sharpening(self, img, intensity=50):
        """Увеличение резкости изображения без изменения яркости"""
        # Создаем ядро для увеличения резкости
        kernel = np.array(
            [
                [-1, -1, -1],
                [-1, 9, -1],
                [-1, -1, -1],
            ]
        )

        # Применяем фильтр
        if len(img.shape) == 3:
            sharpened = np.zeros_like(img)
            for i in range(3):
                sharpened[:, :, i] = cv2.filter2D(img[:, :, i], -1, kernel)
        else:
            sharpened = cv2.filter2D(img, -1, kernel)

        # Смешиваем с оригиналом в зависимости от интенсивности
        alpha = (intensity / 100.0) * 0.3  # Максимум 30% резкости
        result = cv2.addWeighted(img, 1 - alpha, sharpened, alpha, 0)

        return result

    def denoise_image(self, image_path, intensity=50):
        """Подавление шума на изображении"""
        def process_func(img, intensity):
            # Настраиваем параметры в зависимости от интенсивности
            h = 5 + (intensity / 100.0) * 15  # от 5 до 20
            hColor = 5 + (intensity / 100.0) * 15
            
            # Применяем Non-local Means Denoising
            denoised = cv2.fastNlMeansDenoisingColored(img, None, 
                                                     h=h, hColor=hColor, 
                                                     templateWindowSize=7, 
                                                     searchWindowSize=21)
            
            return denoised
        
        try:
            return self._process_with_size_support(image_path, process_func, intensity)
        except Exception as e:
            print(f"Ошибка в denoise_image: {e}")
            return None
    
    def sharpen_image(self, image_path, intensity=50):
        """Увеличение резкости изображения"""
        def process_func(img, intensity):
            # Применяем улучшенное увеличение резкости
            result = self._apply_sharpening(img, intensity)
            return result
        
        try:
            return self._process_with_size_support(image_path, process_func, intensity)
        except Exception as e:
            print(f"Ошибка в sharpen_image: {e}")
            return None
    
    def enhance_contrast(self, image_path, intensity=50):
        """Улучшение контраста изображения"""
        def process_func(img, intensity):
            # Применяем CLAHE
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Настраиваем CLAHE в зависимости от интенсивности
            clip_limit = 1.0 + (intensity / 100.0) * 3.0  # от 1.0 до 4.0
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Объединяем каналы
            lab = cv2.merge([l, a, b])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            return result
        
        try:
            return self._process_with_size_support(image_path, process_func, intensity)
        except Exception as e:
            print(f"Ошибка в enhance_contrast: {e}")
            return None
    
    def sar_enhancement(self, image_path, intensity=50):
        """Специальное улучшение для радиолокационных снимков"""
        def process_func(img, intensity):
            # 1. Конвертируем в grayscale для SAR обработки
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            
            # 2. Логарифмическое растяжение для темных SAR изображений
            gray_float = gray.astype(np.float32)
            gray_float = np.log(gray_float + 1)  # log(1 + pixel) для избежания log(0)
            
            # 3. Нормализация
            gray_norm = cv2.normalize(gray_float, None, 0, 255, cv2.NORM_MINMAX)
            gray_norm = gray_norm.astype(np.uint8)
            
            # 4. Подавление шума специально для SAR
            denoised = cv2.fastNlMeansDenoising(gray_norm, None, 
                                              h=15, templateWindowSize=7, searchWindowSize=21)
            
            # 5. Улучшение контраста для темных областей
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            
            # 6. Увеличение резкости
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            
            # Смешиваем с оригиналом
            alpha = (intensity / 100.0) * 0.3
            result = cv2.addWeighted(enhanced, 1 - alpha, sharpened, alpha, 0)
            
            # Конвертируем обратно в BGR для отображения
            if len(img.shape) == 3:
                result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            
            return result
        
        try:
            return self._process_with_size_support(image_path, process_func, intensity)
        except Exception as e:
            print(f"Ошибка в sar_enhancement: {e}")
            return None
    
    def sar_denoise(self, image_path, intensity=50):
        """Специальное подавление шума для SAR изображений"""
        def process_func(img, intensity):
            # Конвертируем в grayscale
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            
            # Адаптивное подавление шума для SAR
            # Используем более агрессивные параметры для темных изображений
            h = 10 + (intensity / 100.0) * 20  # от 10 до 30
            
            # Non-local Means Denoising
            denoised = cv2.fastNlMeansDenoising(gray, None, h=h, 
                                              templateWindowSize=7, searchWindowSize=21)
            
            # Дополнительное сглаживание для удаления зернистости
            denoised = cv2.bilateralFilter(denoised, 9, 75, 75)
            
            # Конвертируем обратно в BGR
            if len(img.shape) == 3:
                result = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
            else:
                result = denoised
            
            return result
        
        try:
            return self._process_with_size_support(image_path, process_func, intensity)
        except Exception as e:
            print(f"Ошибка в sar_denoise: {e}")
            return None
    
    def sar_brighten(self, image_path, intensity=50):
        """Осветление темных SAR изображений"""
        def process_func(img, intensity):
            # Конвертируем в float для обработки
            img_float = img.astype(np.float32) / 255.0
            
            # Агрессивное осветление для темных SAR изображений
            brightness_factor = 1.5 + (intensity / 100.0) * 2.0  # от 1.5 до 3.5
            img_bright = np.clip(img_float * brightness_factor, 0, 1)
            
            # Гамма-коррекция для улучшения темных областей
            gamma = 0.5 + (intensity / 100.0) * 0.5  # от 0.5 до 1.0
            img_gamma = np.power(img_bright, gamma)
            
            # Специальная обработка для очень темных пикселей
            dark_mask = img_gamma < 0.1
            img_gamma[dark_mask] = np.power(img_gamma[dark_mask], 0.3)  # Очень ярко
            
            # Конвертируем обратно в uint8
            result = np.clip(img_gamma * 255, 0, 255).astype(np.uint8)
            
            return result
        
        try:
            return self._process_with_size_support(image_path, process_func, intensity)
        except Exception as e:
            print(f"Ошибка в sar_brighten: {e}")
            return None

    def sar_auto_enhance(self, image_path, intensity=50):
        """Автоматическое улучшение SAR снимков с адаптивным выбором параметров"""
        def process_func(img, intensity):
            # Анализируем характеристики изображения
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            
            # Анализ яркости для адаптивного выбора параметров
            mean_brightness = np.mean(gray)
            
            # Адаптивные параметры на основе характеристик изображения
            if mean_brightness < 50:  # Очень темное изображение
                # Агрессивное осветление
                brightness_factor = 2.0 + (intensity / 100.0) * 2.0
                gamma = 0.3 + (intensity / 100.0) * 0.4
                clahe_limit = 4.0
            elif mean_brightness < 100:  # Темное изображение
                # Умеренное осветление
                brightness_factor = 1.5 + (intensity / 100.0) * 1.0
                gamma = 0.5 + (intensity / 100.0) * 0.3
                clahe_limit = 3.0
            else:  # Нормальное изображение
                # Легкое улучшение
                brightness_factor = 1.0 + (intensity / 100.0) * 0.5
                gamma = 0.8 + (intensity / 100.0) * 0.2
                clahe_limit = 2.0
            
            # 1. Логарифмическое растяжение для SAR
            gray_float = gray.astype(np.float32)
            gray_float = np.log(gray_float + 1)
            gray_norm = cv2.normalize(gray_float, None, 0, 255, cv2.NORM_MINMAX)
            gray_norm = gray_norm.astype(np.uint8)
            
            # 2. Подавление шума
            denoised = cv2.fastNlMeansDenoising(gray_norm, None, h=15, 
                                              templateWindowSize=7, searchWindowSize=21)
            
            # 3. Адаптивное осветление
            denoised_float = denoised.astype(np.float32) / 255.0
            brightened = np.clip(denoised_float * brightness_factor, 0, 1)
            
            # 4. Гамма-коррекция
            gamma_corrected = np.power(brightened, gamma)
            
            # 5. CLAHE для улучшения контраста
            gamma_uint8 = (gamma_corrected * 255).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=clahe_limit, tileGridSize=(8, 8))
            enhanced = clahe.apply(gamma_uint8)
            
            # 6. Легкое увеличение резкости
            kernel = np.array([[-0.5, -0.5, -0.5], [-0.5, 5, -0.5], [-0.5, -0.5, -0.5]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            
            # Смешиваем с оригиналом
            alpha = (intensity / 100.0) * 0.2
            result = cv2.addWeighted(enhanced, 1 - alpha, sharpened, alpha, 0)
            
            # Конвертируем обратно в BGR
            if len(img.shape) == 3:
                result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            
            return result
        
        try:
            return self._process_with_size_support(image_path, process_func, intensity)
        except Exception as e:
            print(f"Ошибка в sar_auto_enhance: {e}")
            return None

    def sar_contrast(self, image_path, intensity=50):
        """Специализированное улучшение контраста для SAR изображений"""
        def process_func(img, intensity):
            # Конвертируем в grayscale
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            
            # Адаптивные параметры CLAHE для SAR
            clip_limit = 1.0 + (intensity / 100.0) * 4.0  # от 1.0 до 5.0
            tile_size = 8
            
            # Применяем CLAHE
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
            enhanced = clahe.apply(gray)
            
            # Дополнительное выравнивание гистограммы для темных областей
            if np.mean(gray) < 80:  # Если изображение темное
                # Применяем дополнительное растяжение контраста
                enhanced_float = enhanced.astype(np.float32)
                enhanced_float = enhanced_float / 255.0
                enhanced_float = np.power(enhanced_float, 0.7)  # Гамма-коррекция
                enhanced = (enhanced_float * 255).astype(np.uint8)
            
            # Конвертируем обратно в BGR
            if len(img.shape) == 3:
                result = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            else:
                result = enhanced
            
            return result
        
        try:
            return self._process_with_size_support(image_path, process_func, intensity)
        except Exception as e:
            print(f"Ошибка в sar_contrast: {e}")
            return None

    def sar_sharpen(self, image_path, intensity=50):
        """Специализированное увеличение резкости для SAR изображений"""
        def process_func(img, intensity):
            # Конвертируем в grayscale
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            
            # Специальное ядро для SAR изображений (подчеркивает границы объектов)
            kernel = np.array([[-1, -1, -1, -1, -1],
                              [-1, -1, -1, -1, -1],
                              [-1, -1, 25, -1, -1],
                              [-1, -1, -1, -1, -1],
                              [-1, -1, -1, -1, -1]])
            
            # Применяем фильтр резкости
            sharpened = cv2.filter2D(gray, -1, kernel)
            
            # Адаптивное смешивание с оригиналом
            alpha = (intensity / 100.0) * 0.4  # Максимум 40% резкости
            result = cv2.addWeighted(gray, 1 - alpha, sharpened, alpha, 0)
            
            # Дополнительное подавление шума для предотвращения артефактов
            result = cv2.bilateralFilter(result, 5, 50, 50)
            
            # Конвертируем обратно в BGR
            if len(img.shape) == 3:
                result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            
            return result
        
        try:
            return self._process_with_size_support(image_path, process_func, intensity)
        except Exception as e:
            print(f"Ошибка в sar_sharpen: {e}")
            return None

    def sar_comprehensive(self, image_path, intensity=50):
        """Комплексное улучшение SAR изображений - полный пайплайн"""
        def process_func(img, intensity):
            # Конвертируем в grayscale для обработки
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            
            # 1. Логарифмическое растяжение для SAR
            gray_float = gray.astype(np.float32)
            gray_float = np.log(gray_float + 1)
            gray_norm = cv2.normalize(gray_float, None, 0, 255, cv2.NORM_MINMAX)
            gray_norm = gray_norm.astype(np.uint8)
            
            # 2. Агрессивное подавление шума
            denoised = cv2.fastNlMeansDenoising(gray_norm, None, h=20, 
                                              templateWindowSize=7, searchWindowSize=21)
            
            # 3. Дополнительное сглаживание
            denoised = cv2.bilateralFilter(denoised, 9, 75, 75)
            
            # 4. Адаптивное осветление
            mean_brightness = np.mean(denoised)
            if mean_brightness < 80:
                brightness_factor = 1.5 + (intensity / 100.0) * 1.5
                gamma = 0.4 + (intensity / 100.0) * 0.4
            else:
                brightness_factor = 1.0 + (intensity / 100.0) * 0.8
                gamma = 0.6 + (intensity / 100.0) * 0.3
            
            denoised_float = denoised.astype(np.float32) / 255.0
            brightened = np.clip(denoised_float * brightness_factor, 0, 1)
            gamma_corrected = np.power(brightened, gamma)
            
            # 5. CLAHE для контраста
            gamma_uint8 = (gamma_corrected * 255).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gamma_uint8)
            
            # 6. Увеличение резкости
            kernel = np.array([[-0.5, -0.5, -0.5], [-0.5, 5, -0.5], [-0.5, -0.5, -0.5]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            
            # 7. Финальное смешивание
            alpha = (intensity / 100.0) * 0.3
            result = cv2.addWeighted(enhanced, 1 - alpha, sharpened, alpha, 0)
            
            # 8. Финальная нормализация
            result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
            result = result.astype(np.uint8)
            
            # Конвертируем обратно в BGR
            if len(img.shape) == 3:
                result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            
            return result
        
        try:
            return self._process_with_size_support(image_path, process_func, intensity)
        except Exception as e:
            print(f"Ошибка в sar_comprehensive: {e}")
            return None

    def _initialize_models(self):
        """Инициализация моделей подавления шума"""
        try:
            # Попытка создать простую модель если нет предобученных
            if not self.model_manager.create_simple_denoising_model():
                print("Не удалось создать модель, будет использован fallback")
        except Exception as e:
            print(f"Ошибка инициализации моделей: {e}")

    def sar_ai_denoise(self, image_path, intensity=50):
        """AI-подавление шума для SAR изображений с использованием нейронных сетей"""
        def process_func(img, intensity):
            # Конвертируем в grayscale для обработки
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            
            # Попытка использовать нейронную сеть
            ai_denoised = self._apply_ai_denoising(gray, intensity)
            
            if ai_denoised is not None:
                # Конвертируем обратно в BGR
                if len(img.shape) == 3:
                    result = cv2.cvtColor(ai_denoised, cv2.COLOR_GRAY2BGR)
                else:
                    result = ai_denoised
                return result
            else:
                # Fallback на традиционные методы
                print("AI модель недоступна, используем fallback методы")
                fallback_result = self._apply_fallback_denoising(gray, intensity)
                if len(img.shape) == 3:
                    return cv2.cvtColor(fallback_result, cv2.COLOR_GRAY2BGR)
                return fallback_result
        
        try:
            return self._process_with_size_support(image_path, process_func, intensity)
        except Exception as e:
            print(f"Ошибка в sar_ai_denoise: {e}")
            return None

    def sar_ai_enhance(self, image_path, intensity=50):
        """Комплексное AI-улучшение SAR изображений"""
        def process_func(img, intensity):
            # Конвертируем в grayscale
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            
            # 1. AI-подавление шума
            denoised = self._apply_ai_denoising(gray, intensity)
            if denoised is None:
                denoised = self._apply_fallback_denoising(gray, intensity)
            
            # 2. Адаптивное осветление
            mean_brightness = np.mean(denoised)
            if mean_brightness < 80:
                brightness_factor = 1.5 + (intensity / 100.0) * 1.5
                gamma = 0.4 + (intensity / 100.0) * 0.4
            else:
                brightness_factor = 1.0 + (intensity / 100.0) * 0.8
                gamma = 0.6 + (intensity / 100.0) * 0.3
            
            # Применяем осветление
            denoised_float = denoised.astype(np.float32) / 255.0
            brightened = np.clip(denoised_float * brightness_factor, 0, 1)
            gamma_corrected = np.power(brightened, gamma)
            
            # 3. CLAHE для контраста
            gamma_uint8 = (gamma_corrected * 255).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gamma_uint8)
            
            # 4. Легкое увеличение резкости
            kernel = np.array([[-0.5, -0.5, -0.5], [-0.5, 5, -0.5], [-0.5, -0.5, -0.5]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            
            # Смешиваем с оригиналом
            alpha = (intensity / 100.0) * 0.3
            result = cv2.addWeighted(enhanced, 1 - alpha, sharpened, alpha, 0)
            
            # Конвертируем обратно в BGR
            if len(img.shape) == 3:
                result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            
            return result
        
        try:
            return self._process_with_size_support(image_path, process_func, intensity)
        except Exception as e:
            print(f"Ошибка в sar_ai_enhance: {e}")
            return None

    def _apply_ai_denoising(self, image, intensity):
        """Применение AI модели для подавления шума"""
        try:
            # Получаем доступные модели
            available_models = self.model_manager.list_available_models()
            
            if not available_models:
                return None
            
            # Используем первую доступную модель
            model_name = available_models[0]
            model = self.model_manager.get_model(model_name)
            
            if model is None:
                return None
            
            # Применяем модель
            result = model.denoise(image)
            return result
            
        except Exception as e:
            print(f"Ошибка применения AI модели: {e}")
            return None

    def _apply_fallback_denoising(self, image, intensity):
        """Применение fallback методов подавления шума"""
        try:
            # Комбинируем несколько методов для лучшего результата
            # 1. Bilateral фильтр
            bilateral = self.fallback_denoising.bilateral_denoise(image, intensity)
            
            # 2. NLM denoising
            nlm = self.fallback_denoising.nlm_denoise(image, intensity)
            
            # 3. Смешиваем результаты
            alpha = 0.6  # Больше веса для NLM
            result = cv2.addWeighted(bilateral, 1 - alpha, nlm, alpha, 0)
            
            return result
            
        except Exception as e:
            print(f"Ошибка в fallback denoising: {e}")
            return image

    def sar_morphological_enhance(self, image_path, intensity=50):
        """Новый алгоритм улучшения SAR снимков на основе морфологических операций"""
        def process_func(img, intensity):
            # Конвертируем в grayscale для обработки
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            
            # Анализ характеристик изображения для адаптивных параметров
            mean_brightness = np.mean(gray)
            intensity_factor = intensity / 100.0
            
            # 1. Логарифмическое растяжение для темных SAR изображений
            gray_float = gray.astype(np.float32)
            gray_float = np.log(gray_float + 1)
            gray_norm = cv2.normalize(gray_float, None, 0, 255, cv2.NORM_MINMAX)
            gray_norm = gray_norm.astype(np.uint8)
            
            # 2. Морфологическое открытие для удаления мелкого шума
            kernel_size = int(3 + intensity_factor * 4)  # от 3 до 7
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            opened = cv2.morphologyEx(gray_norm, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # 3. Морфологическое закрытие для заполнения мелких отверстий
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            # 4. Адаптивная фильтрация для сохранения деталей
            d = int(5 + intensity_factor * 4)
            sigma_color = 50 + intensity_factor * 50
            sigma_space = 50 + intensity_factor * 50
            bilateral = cv2.bilateralFilter(closed, d, sigma_color, sigma_space)
            
            # 5. Комбинируем результаты
            morph_weight = 0.3 + (1 - intensity_factor) * 0.3
            bilateral_weight = 1.0 - morph_weight
            combined = cv2.addWeighted(closed, morph_weight, bilateral, bilateral_weight, 0)
            
            # 6. Адаптивное улучшение контраста
            if mean_brightness < 60:
                clahe_limit = 2.0 + intensity_factor * 3.0
            elif mean_brightness < 100:
                clahe_limit = 1.5 + intensity_factor * 2.5
            else:
                clahe_limit = 1.0 + intensity_factor * 2.0
            
            clahe = cv2.createCLAHE(clipLimit=clahe_limit, tileGridSize=(8, 8))
            enhanced = clahe.apply(combined)
            
            # 7. Селективное усиление границ объектов
            gradient = cv2.morphologyEx(enhanced, cv2.MORPH_GRADIENT, kernel)
            edge_strength = intensity_factor * 0.2
            enhanced_with_edges = cv2.addWeighted(enhanced, 1.0 - edge_strength, gradient, edge_strength, 0)
            
            # 8. Финальная адаптивная коррекция яркости
            mean_enhanced = np.mean(enhanced_with_edges)
            if mean_enhanced < 80:
                gamma = 0.5 + intensity_factor * 0.3
                enhanced_float = enhanced_with_edges.astype(np.float32) / 255.0
                gamma_corrected = np.power(enhanced_float, gamma)
                final = (gamma_corrected * 255).astype(np.uint8)
            else:
                final = enhanced_with_edges
            
            # 9. Финальная нормализация
            final = cv2.normalize(final, None, 0, 255, cv2.NORM_MINMAX)
            final = final.astype(np.uint8)
            
            # Конвертируем обратно в BGR
            if len(img.shape) == 3:
                result = cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)
            else:
                result = final
            
            return result
        
        try:
            return self._process_with_size_support(image_path, process_func, intensity)
        except Exception as e:
            print(f"Ошибка в sar_morphological_enhance: {e}")
            return None

    def enhance_image(self, image_path, method='baseline', intensity=50):
        """Главная функция улучшения изображения"""
        try:
            if method not in self.enhancement_methods:
                raise ValueError(f"Неизвестный метод: {method}")
            
            # Применяем выбранный метод
            enhanced_img = self.enhancement_methods[method](image_path, intensity)
            
            if enhanced_img is None:
                return None, None
            
            # Загружаем оригинальное изображение для сравнения
            original_img, _, _, _ = self._load_image_any_size(image_path)
            if original_img is None:
                original_img = cv2.imread(image_path)
            
            if original_img is None:
                return enhanced_img, None
            
            # Вычисляем метрики качества
            metrics = self.calculate_quality_metrics(original_img, enhanced_img)
            
            return enhanced_img, metrics
            
        except Exception as e:
            print(f"Ошибка в enhance_image: {e}")
            return None, None

    def calculate_quality_metrics(self, original_img, enhanced_img):
        """Вычисление метрик качества улучшения"""
        try:
            # Конвертируем в grayscale для вычислений
            if len(original_img.shape) == 3:
                orig_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
                enh_gray = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
            else:
                orig_gray = original_img
                enh_gray = enhanced_img
            
            # PSNR
            mse = np.mean((orig_gray.astype(float) - enh_gray.astype(float)) ** 2)
            if mse == 0:
                psnr = float('inf')
            else:
                psnr = 20 * np.log10(255.0 / np.sqrt(mse))
            
            # Контраст
            contrast_orig = np.std(orig_gray)
            contrast_enh = np.std(enh_gray)
            contrast_improvement = (contrast_enh - contrast_orig) / contrast_orig * 100
            
            # Яркость
            brightness_orig = np.mean(orig_gray)
            brightness_enh = np.mean(enh_gray)
            brightness_change = (brightness_enh - brightness_orig) / brightness_orig * 100
            
            return {
                'psnr': round(psnr, 2),
                'contrast_improvement': round(contrast_improvement, 2),
                'brightness_change': round(brightness_change, 2),
                'original_contrast': round(contrast_orig, 2),
                'enhanced_contrast': round(contrast_enh, 2)
            }
            
        except Exception as e:
            print(f"Ошибка в calculate_quality_metrics: {e}")
            return {
                'psnr': 0,
                'contrast_improvement': 0,
                'brightness_change': 0,
                'original_contrast': 0,
                'enhanced_contrast': 0
            }

    def save_enhanced_image(self, enhanced_img, output_path):
        """Сохранение улучшенного изображения"""
        try:
            if enhanced_img is not None:
                cv2.imwrite(output_path, enhanced_img)
                return True
            return False
        except Exception as e:
            print(f"Ошибка при сохранении: {e}")
            return False


__all__ = ["ImageEnhancement"]



