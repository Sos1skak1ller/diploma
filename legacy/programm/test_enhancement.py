#!/usr/bin/env python3
"""
Тестовый скрипт для алгоритма улучшения изображений
"""

import sys
import os

def test_enhancement():
    """Тестирование алгоритма улучшения"""
    
    print("Тестирование алгоритма улучшения изображений...")
    
    try:
        # Проверяем импорт OpenCV
        import cv2
        print(f"✓ OpenCV версия: {cv2.__version__}")
    except ImportError:
        print("✗ OpenCV не установлен. Установите: pip install opencv-python")
        return False
    
    try:
        # Проверяем импорт numpy
        import numpy as np
        print(f"✓ NumPy версия: {np.__version__}")
    except ImportError:
        print("✗ NumPy не установлен. Установите: pip install numpy")
        return False
    
    try:
        # Проверяем импорт PIL
        from PIL import Image
        print("✓ Pillow импортирован успешно")
    except ImportError:
        print("✗ Pillow не установлен. Установите: pip install pillow")
        return False
    
    try:
        # Импортируем наш алгоритм
        from src.models.image_enhancement import ImageEnhancement
        print("✓ Алгоритм улучшения импортирован успешно")
    except ImportError as e:
        print(f"✗ Ошибка импорта алгоритма: {e}")
        return False
    
    # Тестируем с тестовым изображением
    test_img = 'images/1.jpg'
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
        enhanced, metrics = enhancer.enhance_image(test_img, 'baseline', 50)
        
        if enhanced is not None:
            print("✓ Baseline улучшение работает")
            print(f"Метрики качества: {metrics}")
            
            # Сохраняем результат
            output_path = 'test_enhanced.jpg'
            success = enhancer.save_enhanced_image(enhanced, output_path)
            
            if success:
                print(f"✓ Улучшенное изображение сохранено: {output_path}")
            else:
                print("✗ Ошибка сохранения улучшенного изображения")
        else:
            print("✗ Ошибка в baseline улучшении")
            return False
            
    except Exception as e:
        print(f"✗ Ошибка при тестировании алгоритма: {e}")
        return False
    
    print("\n✅ Все тесты пройдены успешно!")
    print("Алгоритм улучшения готов к использованию.")
    return True

if __name__ == '__main__':
    success = test_enhancement()
    sys.exit(0 if success else 1)
