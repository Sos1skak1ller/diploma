#!/usr/bin/env python3
"""
Тестовый скрипт для проверки SAR улучшения изображений
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from src.models.image_enhancement import ImageEnhancement
import cv2
import numpy as np

def test_sar_enhancement():
    """Тестирование SAR методов улучшения"""
    
    # Создаем экземпляр улучшения
    enhancer = ImageEnhancement()
    
    # Тестовые изображения
    test_images = [
        "images/1.jpg",
        "images/9.jpg"
    ]
    
    print("Тестирование SAR методов улучшения...")
    
    for image_path in test_images:
        if not os.path.exists(image_path):
            print(f"Изображение {image_path} не найдено, пропускаем...")
            continue
            
        print(f"\nОбработка: {image_path}")
        
        # Тестируем различные SAR методы
        methods = [
            'sar_auto_enhance',
            'sar_denoise', 
            'sar_brighten',
            'sar_contrast',
            'sar_sharpen',
            'sar_comprehensive'
        ]
        
        for method in methods:
            try:
                print(f"  Тестирование метода: {method}")
                enhanced_img, metrics = enhancer.enhance_image(image_path, method=method, intensity=70)
                
                if enhanced_img is not None:
                    print(f"    ✓ Успешно обработано")
                    if metrics:
                        print(f"    PSNR: {metrics.get('psnr', 0):.2f} dB")
                        print(f"    Улучшение контраста: {metrics.get('contrast_improvement', 0):.1f}%")
                else:
                    print(f"    ✗ Ошибка обработки")
                    
            except Exception as e:
                print(f"    ✗ Ошибка: {e}")
    
    print("\nТестирование завершено!")

if __name__ == "__main__":
    test_sar_enhancement()
