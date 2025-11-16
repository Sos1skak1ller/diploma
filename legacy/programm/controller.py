"""
Контроллер для интеграции с существующим кодом
"""

class Controller:
    """Базовый контроллер"""
    
    def __init__(self, view):
        self.view = view
        
    def set_ui_enabled(self, enabled):
        """Включение/отключение элементов интерфейса"""
        pass
        
    def update_progress(self, progress, filename=""):
        """Обновление прогресса"""
        pass
