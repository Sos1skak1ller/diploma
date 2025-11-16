#!/usr/bin/env python3
"""
Тестовый скрипт для проверки AI методов подавления шума (новый путь импорта)
"""

import os

from main.algorythms.improvment.image_enhancement import ImageEnhancement
from main.algorythms.improvment.denoising_models import DenoisingModelManager
import cv2


def test_ai_denoising():
    """Тестирование AI методов подавления шума"""

    print("Тестирование AI методов подавления шума...")

    # Создаем экземпляр улучшения
    enhancer = ImageEnhancement()

    # Тестовые изображения (ожидаем, что они лежат в images/ рядом с проектом)
    base = "images"
    test_images = [
        os.path.join(base, "1.jpg"),
        os.path.join(base, "9.jpg"),
    ]

    # Тестируем AI методы
    ai_methods = [
        "sar_ai_denoise",
        "sar_ai_enhance",
    ]

    for image_path in test_images:
        if not os.path.exists(image_path):
            print(f"Изображение {image_path} не найдено, пропускаем...")
            continue

        print(f"\nОбработка: {image_path}")

        for method in ai_methods:
            try:
                print(f"  Тестирование AI метода: {method}")
                enhanced_img, metrics = enhancer.enhance_image(
                    image_path, method=method, intensity=70
                )

                if enhanced_img is not None:
                    print("    ✓ Успешно обработано")
                    if metrics:
                        print(f"    PSNR: {metrics.get('psnr', 0):.2f} dB")
                        print(
                            f"    Улучшение контраста: {metrics.get('contrast_improvement', 0):.1f}%"
                        )

                    # Сохраняем результат в tmp/
                    os.makedirs("tmp", exist_ok=True)
                    output_path = os.path.join(
                        "tmp",
                        f"test_output_{method}_{os.path.basename(image_path)}",
                    )
                    cv2.imwrite(output_path, enhanced_img)
                    print(f"    Результат сохранен: {output_path}")
                else:
                    print("    ✗ Ошибка обработки")

            except Exception as e:
                print(f"    ✗ Ошибка: {e}")

    print("\nТестирование AI методов завершено!")


def test_model_manager():
    """Тестирование менеджера моделей"""

    print("\nТестирование менеджера моделей...")

    manager = DenoisingModelManager()

    # Список доступных моделей
    models = manager.list_available_models()
    print(f"Доступные модели: {models}")

    # Попытка создать простую модель
    print("Создание простой модели...")
    success = manager.create_simple_denoising_model()
    if success:
        print("✓ Простая модель создана")
    else:
        print("✗ Не удалось создать модель")

    # Проверяем доступные модели после создания
    models = manager.list_available_models()
    print(f"Модели после создания: {models}")


if __name__ == "__main__":
    test_ai_denoising()
    test_model_manager()


