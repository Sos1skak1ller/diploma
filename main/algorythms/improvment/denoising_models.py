"""
Легковесные модели для подавления шума на спутниковых снимках.
Полная реализация перенесена в пакет main, чтобы не зависеть от других директорий.
"""

import os
import cv2
import numpy as np
from typing import Optional
import requests
from pathlib import Path
import torch
import torch.nn as nn

try:  # onnxruntime опционален: нужен только для AI-денойзинга
    import onnxruntime as ort  # type: ignore[import]
except Exception:
    ort = None


class LightweightDenoisingModel:
    """Базовый класс для легковесных моделей подавления шума"""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.session = None
        self.device = "cpu"
        self.input_name = None
        self.output_name = None

    def load_model(self, model_path: str) -> bool:
        """Загрузка ONNX модели"""
        try:
            if ort is None:
                print("onnxruntime не установлен, пропускаем загрузку ONNX модели")
                return False
            if not os.path.exists(model_path):
                print(f"Модель не найдена: {model_path}")
                return False

            # Создаем сессию ONNX Runtime
            providers = ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(model_path, providers=providers)

            # Получаем имена входов и выходов
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name

            print(f"Модель загружена: {model_path}")
            return True

        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            return False

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Предобработка изображения для модели"""
        # Нормализация в диапазон [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        # Добавляем batch и channel размерности если нужно
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=0)  # channel
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)  # batch

        return image.astype(np.float32)

    def postprocess_image(self, output: np.ndarray) -> np.ndarray:
        """Постобработка результата модели"""
        # Убираем batch размерность
        if len(output.shape) == 4:
            output = output.squeeze(0)
        if len(output.shape) == 3:
            output = output.squeeze(0)

        # Нормализация в диапазон [0, 255]
        output = np.clip(output, 0, 1)
        output = (output * 255).astype(np.uint8)

        return output

    def denoise(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Применение модели для подавления шума"""
        if self.session is None:
            print("Модель не загружена")
            return None

        try:
            # Предобработка
            input_data = self.preprocess_image(image)

            # Инференс
            outputs = self.session.run([self.output_name], {self.input_name: input_data})
            output = outputs[0]

            # Постобработка
            result = self.postprocess_image(output)

            return result

        except Exception as e:
            print(f"Ошибка при применении модели: {e}")
            return None


class DnCNNModel(LightweightDenoisingModel):
    """DnCNN модель для подавления шума"""

    def __init__(self):
        super().__init__()
        self.model_name = "DnCNN"

    def create_simple_dncnn(self, input_channels: int = 1, num_layers: int = 17) -> nn.Module:
        """Создание простой DnCNN архитектуры"""

        class SimpleDnCNN(nn.Module):
            def __init__(self, channels=1, num_of_layers=17):
                super(SimpleDnCNN, self).__init__()
                kernel_size = 3
                padding = 1
                features = 64
                layers = []

                # Первый слой
                layers.append(
                    nn.Conv2d(
                        in_channels=channels,
                        out_channels=features,
                        kernel_size=kernel_size,
                        padding=padding,
                        bias=False,
                    )
                )
                layers.append(nn.ReLU(inplace=True))

                # Скрытые слои
                for _ in range(num_of_layers - 2):
                    layers.append(
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=kernel_size,
                            padding=padding,
                            bias=False,
                        )
                    )
                    layers.append(nn.BatchNorm2d(features))
                    layers.append(nn.ReLU(inplace=True))

                # Последний слой
                layers.append(
                    nn.Conv2d(
                        in_channels=features,
                        out_channels=channels,
                        kernel_size=kernel_size,
                        padding=padding,
                        bias=False,
                    )
                )

                self.dncnn = nn.Sequential(*layers)

            def forward(self, x):
                out = self.dncnn(x)
                return out

        return SimpleDnCNN(input_channels, num_layers)


class RIDNetModel(LightweightDenoisingModel):
    """RIDNet модель для подавления шума"""

    def __init__(self):
        super().__init__()
        self.model_name = "RIDNet"


class CBDNetModel(LightweightDenoisingModel):
    """CBDNet модель для подавления шума"""

    def __init__(self):
        super().__init__()
        self.model_name = "CBDNet"


class DenoisingModelManager:
    """Менеджер для управления моделями подавления шума"""

    def __init__(self):
        self.models = {}
        # Храним файлы моделей рядом с пакетом main
        root = Path(__file__).resolve().parents[3]  # .../diplom
        self.models_dir = root / "models" / "denoising"
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def register_model(self, name: str, model: LightweightDenoisingModel):
        """Регистрация модели"""
        self.models[name] = model

    def get_model(self, name: str) -> Optional[LightweightDenoisingModel]:
        """Получение модели по имени"""
        return self.models.get(name)

    def list_available_models(self) -> list:
        """Список доступных моделей"""
        return list(self.models.keys())

    def download_pretrained_model(self, model_name: str, url: str) -> bool:
        """Загрузка предобученной модели"""
        try:
            model_path = self.models_dir / f"{model_name}.onnx"

            if model_path.exists():
                print(f"Модель {model_name} уже существует")
                return True

            print(f"Загрузка модели {model_name}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"Модель {model_name} загружена: {model_path}")
            return True

        except Exception as e:
            print(f"Ошибка загрузки модели {model_name}: {e}")
            return False

    def create_simple_denoising_model(self, model_name: str = "simple_dncnn") -> bool:
        """Создание простой модели подавления шума"""
        try:
            # Создаем простую DnCNN модель
            dncnn = DnCNNModel()
            model = dncnn.create_simple_dncnn()

            # Сохраняем модель
            model_path = self.models_dir / f"{model_name}.pth"
            torch.save(model.state_dict(), model_path)

            # Конвертируем в ONNX
            model.eval()
            dummy_input = torch.randn(1, 1, 256, 256)
            onnx_path = self.models_dir / f"{model_name}.onnx"

            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch_size", 2: "height", 3: "width"},
                    "output": {0: "batch_size", 2: "height", 3: "width"},
                },
            )

            print(f"Простая модель создана: {onnx_path}")
            return True

        except Exception as e:
            print(f"Ошибка создания модели: {e}")
            return False


def create_fallback_denoising():
    """Создание fallback методов подавления шума"""

    class FallbackDenoising:
        """Fallback методы подавления шума без нейронных сетей"""

        @staticmethod
        def bilateral_denoise(image: np.ndarray, intensity: int = 50) -> np.ndarray:
            """Билатеральный фильтр для подавления шума"""
            d = 5 + (intensity // 20) * 5  # от 5 до 25
            sigma_color = 50 + intensity * 2  # от 50 до 150
            sigma_space = 50 + intensity * 2

            return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

        @staticmethod
        def nlm_denoise(image: np.ndarray, intensity: int = 50) -> np.ndarray:
            """Non-local Means Denoising"""
            h = 5 + (intensity / 100.0) * 15  # от 5 до 20

            if len(image.shape) == 3:
                return cv2.fastNlMeansDenoisingColored(
                    image, None, h=h, hColor=h, templateWindowSize=7, searchWindowSize=21
                )
            return cv2.fastNlMeansDenoising(
                image, None, h=h, templateWindowSize=7, searchWindowSize=21
            )

        @staticmethod
        def wavelet_denoise(image: np.ndarray, intensity: int = 50) -> np.ndarray:
            """Простая вейвлет-подобная фильтрация через медианный фильтр"""
            kernel_size = 3 + (intensity // 25) * 2  # от 3 до 11
            if kernel_size % 2 == 0:
                kernel_size += 1

            if len(image.shape) == 3:
                result = np.zeros_like(image)
                for i in range(3):
                    result[:, :, i] = cv2.medianBlur(image[:, :, i], kernel_size)
                return result
            return cv2.medianBlur(image, kernel_size)

    return FallbackDenoising()


__all__ = [
    "LightweightDenoisingModel",
    "DnCNNModel",
    "RIDNetModel",
    "CBDNetModel",
    "DenoisingModelManager",
    "create_fallback_denoising",
]



