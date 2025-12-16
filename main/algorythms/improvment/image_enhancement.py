"""
Упрощённый алгоритм улучшения SAR-снимков.

Оставлен только один надёжный пайплайн подавления шума (fallback),
который всегда используется из интерфейса.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional


class ImageEnhancement:
    """
    Минимальный класс для улучшения качества радиолокационных (SAR) снимков.

    В текущей версии реализованы лёгкие CPU‑алгоритмы:
    - гибридное подавление шума (bilateral + NLM, fallback‑пайплайн);
    - адаптивный фильтр спекла (локальная статистика, Lee/Frost‑подобный);
    - анизотропная диффузия (Perona–Malik‑подобный SRAD для SAR).
    """

    def _apply_fallback_denoising(self, image: np.ndarray, intensity: int) -> np.ndarray:
        """Применение fallback методов подавления шума.

        Логика совпадает с прежним fallback‑пайплайном:
        1. Bilateral фильтр.
        2. Non-local Means (NLM) denoising.
        3. Смешивание результатов.
        """
        try:
            # 1. Bilateral фильтр
            d = 5 + (intensity // 20) * 5  # от 5 до 25
            sigma_color = 50 + intensity * 2  # от 50 до 150
            sigma_space = 50 + intensity * 2

            bilateral = cv2.bilateralFilter(image, d, sigma_color, sigma_space)

            # 2. NLM denoising
            h = 5 + (intensity / 100.0) * 15  # от 5 до 20
            if len(image.shape) == 3:
                nlm = cv2.fastNlMeansDenoisingColored(
                    image, None, h=h, hColor=h, templateWindowSize=7, searchWindowSize=21
                )
            else:
                nlm = cv2.fastNlMeansDenoising(
                    image, None, h=h, templateWindowSize=7, searchWindowSize=21
                )

            # 3. Смешиваем результаты
            alpha = 0.6  # Больше веса для NLM
            result = cv2.addWeighted(bilateral, 1 - alpha, nlm, alpha, 0)

            return result

        except Exception as e:
            print(f"Ошибка в fallback denoising: {e}")
            return image

    def _adaptive_speckle_filter(self, image: np.ndarray, intensity: int) -> np.ndarray:
        """
        Адаптивный фильтр спекла (Lee/Frost‑подобный).

        Использует локальную статистику в окне:
        - оценивает локальное среднее и дисперсию;
        - подавляет шум, сохраняя контраст на границах.
        """
        try:
            img = image.astype(np.float32)
            # Размер окна в зависимости от интенсивности (3..9)
            win = 3 + 2 * int(intensity / 35)  # 3, 5, 7, 9
            win = max(3, min(9, win))
            ksize = (win, win)

            def _process_channel(ch: np.ndarray) -> np.ndarray:
                mean = cv2.boxFilter(ch, ddepth=-1, ksize=ksize, normalize=True, borderType=cv2.BORDER_REFLECT)
                mean_sq = cv2.boxFilter(
                    ch * ch, ddepth=-1, ksize=ksize, normalize=True, borderType=cv2.BORDER_REFLECT
                )
                var = mean_sq - mean * mean
                var = np.maximum(var, 0.0)

                # Оценка дисперсии шума как медианы локальных дисперсий
                noise_var = np.median(var)
                if noise_var <= 0:
                    return ch

                gain = var / (var + noise_var + 1e-8)
                out = mean + gain * (ch - mean)
                return out

            if img.ndim == 2:
                out = _process_channel(img)
            else:
                out = np.zeros_like(img)
                for c in range(img.shape[2]):
                    out[:, :, c] = _process_channel(img[:, :, c])

            out = np.clip(out, 0.0, 255.0).astype(np.uint8)
            return out

        except Exception as e:
            print(f"Ошибка в adaptive speckle filter: {e}")
            return image

    def _anisotropic_diffusion(self, image: np.ndarray, intensity: int) -> np.ndarray:
        """
        Анизотропная диффузия (Perona–Malik‑подобный SRAD).

        Итеративно сглаживает однородные области и бережно относится к границам.
        Число итераций и коэффициенты зависят от интенсивности.
        """
        try:
            # Работаем в grayscale
            if image.ndim == 3:
                img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
            else:
                img = image.astype(np.float32)

            # Параметры диффузии
            num_iter = 5 + int(intensity / 20)  # от 5 до 10
            num_iter = max(3, min(12, num_iter))
            k = 15.0 + (intensity / 100.0) * 25.0  # порог градиента
            lam = 0.2  # шаг по времени (должен быть < 0.25 для устойчивости)

            for _ in range(num_iter):
                north = np.roll(img, -1, axis=0) - img
                south = np.roll(img, 1, axis=0) - img
                east = np.roll(img, -1, axis=1) - img
                west = np.roll(img, 1, axis=1) - img

                cN = np.exp(-(north / k) ** 2)
                cS = np.exp(-(south / k) ** 2)
                cE = np.exp(-(east / k) ** 2)
                cW = np.exp(-(west / k) ** 2)

                img += lam * (cN * north + cS * south + cE * east + cW * west)

            diffused = np.clip(img, 0.0, 255.0).astype(np.uint8)

            if image.ndim == 3:
                diffused = cv2.cvtColor(diffused, cv2.COLOR_GRAY2BGR)
            return diffused

        except Exception as e:
            print(f"Ошибка в anisotropic diffusion: {e}")
            return image

    def enhance_image(
        self,
        image_path: str,
        method: str = "sar_ai_denoise",
        intensity: int = 50,
    ) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
        """
        Главная функция улучшения изображения.

        Параметр `method` определяет используемый алгоритм:
        - 'sar_ai_denoise' или 'hybrid_sar_denoise' — гибридный fallback (bilateral + NLM);
        - 'sar_adaptive' — адаптивный фильтр спекла (Lee/Frost‑подобный);
        - 'sar_srad' — анизотропная диффузия для SAR.
        При неизвестном методе используется fallback‑пайплайн.
        """
        try:
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Не удалось загрузить изображение: {image_path}")

            if method in ("sar_ai_denoise", "hybrid_sar_denoise"):
                enhanced_img = self._apply_fallback_denoising(img, intensity)
            elif method == "sar_adaptive":
                enhanced_img = self._adaptive_speckle_filter(img, intensity)
            elif method == "sar_srad":
                enhanced_img = self._anisotropic_diffusion(img, intensity)
            else:
                print(f"Предупреждение: неизвестный метод '{method}', используется fallback‑пайплайн")
                enhanced_img = self._apply_fallback_denoising(img, intensity)

            if enhanced_img is None:
                return None, None

            metrics = self._calculate_quality_metrics(img, enhanced_img)
            return enhanced_img, metrics

        except Exception as e:
            print(f"Ошибка в enhance_image: {e}")
            return None, None

    def _calculate_quality_metrics(
        self, original_img: np.ndarray, enhanced_img: np.ndarray
    ) -> Dict[str, Any]:
        """Простые метрики качества улучшения (PSNR, контраст, яркость)."""
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
                psnr = float("inf")
            else:
                psnr = 20 * np.log10(255.0 / np.sqrt(mse))

            # Контраст
            contrast_orig = np.std(orig_gray)
            contrast_enh = np.std(enh_gray)
            contrast_improvement = (
                (contrast_enh - contrast_orig) / contrast_orig * 100
                if contrast_orig != 0
                else 0.0
            )

            # Яркость
            brightness_orig = np.mean(orig_gray)
            brightness_enh = np.mean(enh_gray)
            brightness_change = (
                (brightness_enh - brightness_orig) / brightness_orig * 100
                if brightness_orig != 0
                else 0.0
            )

            return {
                "psnr": round(float(psnr), 2) if psnr != float("inf") else float("inf"),
                "contrast_improvement": round(float(contrast_improvement), 2),
                "brightness_change": round(float(brightness_change), 2),
                "original_contrast": round(float(contrast_orig), 2),
                "enhanced_contrast": round(float(contrast_enh), 2),
            }

        except Exception as e:
            print(f"Ошибка в calculate_quality_metrics: {e}")
            return {
                "psnr": 0.0,
                "contrast_improvement": 0.0,
                "brightness_change": 0.0,
                "original_contrast": 0.0,
                "enhanced_contrast": 0.0,
            }

    def save_enhanced_image(self, enhanced_img: np.ndarray, output_path: str) -> bool:
        """Сохранение улучшенного изображения."""
        try:
            if enhanced_img is not None:
                cv2.imwrite(output_path, enhanced_img)
                return True
            return False
        except Exception as e:
            print(f"Ошибка при сохранении: {e}")
            return False


__all__ = ["ImageEnhancement"]


