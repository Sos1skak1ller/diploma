#!/usr/bin/env python3
"""
Тестовый скрипт для алгоритма улучшения изображений (новый путь импорта)
"""

import os


def test_enhancement():
    """Тестирование алгоритма улучшения"""

    print("Тестирование алгоритма улучшения изображений...")

    try:
        # Проверяем импорт OpenCV
        import cv2  # noqa: F401

        print("✓ OpenCV импортирован успешно")
    except ImportError:
        print("✗ OpenCV не установлен. Установите: pip install opencv-python")
        return False

    try:
        # Проверяем импорт numpy
        import numpy as np  # noqa: F401

        print("✓ NumPy импортирован успешно")
    except ImportError:
        print("✗ NumPy не установлен. Установите: pip install numpy")
        return False

    try:
        # Проверяем импорт PIL
        from PIL import Image  # noqa: F401

        print("✓ Pillow импортирован успешно")
    except ImportError:
        print("✗ Pillow не установлен. Установите: pip install pillow")
        return False

    try:
        # Импортируем наш алгоритм по новому пути
        from main.algorythms.improvment.image_enhancement import ImageEnhancement

        print("✓ Алгоритм улучшения импортирован успешно")
    except ImportError as e:
        print(f"✗ Ошибка импорта алгоритма: {e}")
        return False

    # Тестируем с тестовым изображением (ожидаем, что оно лежит в images/1.jpg рядом с проектом)
    test_img = os.path.join("images", "1.jpg")
    if not os.path.exists(test_img):
        print(f"✗ Тестовое изображение не найдено: {test_img}")
        return False

    print(f"✓ Тестовое изображение найдено: {test_img}")

    try:
        # Создаем экземпляр алгоритма
        enhancer = ImageEnhancement()
        print("✓ Экземпляр алгоритма создан")

        # Тестируем baseline улучшение
        print("Тестирование baseline улучшения...")
        enhanced, metrics = enhancer.enhance_image(test_img, "baseline", 50)

        if enhanced is not None:
            print("✓ Baseline улучшение работает")
            print(f"Метрики качества: {metrics}")
        else:
            print("✗ Ошибка в baseline улучшении")
            return False

    except Exception as e:
        print(f"✗ Ошибка при тестировании алгоритма: {e}")
        return False

    print("\n✅ Все тесты пройдены успешно!")
    print("Алгоритм улучшения готов к использованию.")
    return True


if __name__ == "__main__":
    success = test_enhancement()
    raise SystemExit(0 if success else 1)


