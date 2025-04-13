#!/usr/bin/env python3
"""
Приложение "АГПК из пикселей" с шейдерами для Nuitka
"""
# Сначала импортируем патч для Taichi
try:
    import fix_taichi_path
    print("Патч для Taichi успешно загружен")
except ImportError as e:
    print(f"Предупреждение: не удалось загрузить модуль fix_taichi_path: {e}")

# Импортируем основные библиотеки
import os
import sys
import math
import taichi as ti
import numpy as np

# Используем CPU архитектуру для максимальной совместимости
ti.init(arch=ti.cpu)

# Размеры окна и пикселей
window_width, window_height = 800, 600
pixel_size = 20  # Размер одного пикселя

# Создаем цветовое поле
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(window_width, window_height))

# Время для анимации
time = ti.field(dtype=ti.f32, shape=())

# Определение букв "АГПК" в виде пиксельных матриц
# 1 - пиксель включен, 0 - выключен
letter_A = [
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [1, 1, 1, 1],
    [1, 0, 0, 1],
    [1, 0, 0, 1]
]

letter_G = [
    [0, 1, 1, 1],
    [1, 0, 0, 0],
    [1, 0, 1, 1],
    [1, 0, 0, 1],
    [0, 1, 1, 0]
]

letter_P = [
    [1, 1, 1, 0],
    [1, 0, 0, 1],
    [1, 1, 1, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0]
]

letter_K = [
    [1, 0, 0, 1],
    [1, 0, 1, 0],
    [1, 1, 0, 0],
    [1, 0, 1, 0],
    [1, 0, 0, 1]
]

# Очистка экрана тёмно-серым цветом
@ti.kernel
def clear_screen():
    for i, j in pixels:
        # Создаем градиентный фон
        bg_color = ti.Vector([0.05, 0.05, 0.1]) + ti.Vector([0.0, 0.0, 0.05]) * ti.sin(0.01 * i + time[None])
        pixels[i, j] = bg_color

# Функция для вычисления расстояния от точки до центра буквы
@ti.func
def distance_to_center(x: ti.i32, y: ti.i32, center_x: ti.i32, center_y: ti.i32) -> ti.f32:
    dx = x - center_x
    dy = y - center_y
    return ti.sqrt(dx * dx + dy * dy)

# Функция для рисования свечения (bloom эффект)
@ti.func
def apply_bloom(x: ti.i32, y: ti.i32, center_x: ti.i32, center_y: ti.i32, 
                base_color: ti.types.vector(3, ti.f32), intensity: ti.f32) -> ti.types.vector(3, ti.f32):
    dist = distance_to_center(x, y, center_x, center_y)
    # Экспоненциальное затухание от центра
    bloom_factor = ti.exp(-dist * 0.01 * intensity)
    return base_color * (1.0 + bloom_factor * 0.5)  # Увеличиваем яркость

# Функция для тени
@ti.func
def draw_shadow(x: ti.i32, y: ti.i32, shadow_offset_x: ti.i32, shadow_offset_y: ti.i32):
    shadow_x = x + shadow_offset_x
    shadow_y = y + shadow_offset_y
    
    if 0 <= shadow_x < window_width and 0 <= shadow_y < window_height:
        shadow_color = ti.Vector([0.0, 0.0, 0.0])  # Черная тень
        current_color = pixels[shadow_x, shadow_y]
        shadow_strength = 0.7
        pixels[shadow_x, shadow_y] = current_color * (1 - shadow_strength) + shadow_color * shadow_strength

# Функция для анимации цвета
@ti.func
def animate_color(base_color: ti.types.vector(3, ti.f32), x: ti.i32, y: ti.i32) -> ti.types.vector(3, ti.f32):
    # Пульсирующий эффект
    pulse = 0.2 * ti.sin(time[None] * 2.0 + 0.1 * x + 0.1 * y) + 1.0
    # Радужный эффект
    rainbow_effect = ti.Vector([
        0.5 * ti.sin(time[None] + 0.1 * x) + 0.5,
        0.5 * ti.sin(time[None] + 0.1 * y + 2.0) + 0.5,
        0.5 * ti.sin(time[None] + 0.1 * (x + y) + 4.0) + 0.5
    ])
    # Смешиваем базовый цвет и эффекты
    mix_factor = 0.3  # Сила смешивания эффектов
    result_color = base_color * (1.0 - mix_factor) + rainbow_effect * mix_factor
    return result_color * pulse

# Функция для отрисовки буквы на заданной позиции с шейдерными эффектами
@ti.kernel
def draw_letter_with_effects(letter_matrix: ti.types.ndarray(), start_x: ti.i32, start_y: ti.i32, 
                           base_color: ti.types.vector(3, ti.f32), shadow_offset: ti.i32):
    letter_height = letter_matrix.shape[0]
    letter_width = letter_matrix.shape[1]
    
    # Сначала рисуем тени
    for i in range(letter_height):
        for j in range(letter_width):
            if letter_matrix[i, j] == 1:
                # Центр текущего пикселя
                center_x = start_x + j * pixel_size + pixel_size // 2
                center_y = start_y + i * pixel_size + pixel_size // 2
                
                # Рисуем тень (смещена вниз и вправо)
                for px in range(pixel_size):
                    for py in range(pixel_size):
                        x = start_x + j * pixel_size + px
                        y = start_y + i * pixel_size + py
                        if 0 <= x < window_width and 0 <= y < window_height:
                            draw_shadow(x, y, shadow_offset, shadow_offset)
    
    # Затем рисуем буквы с эффектами
    for i in range(letter_height):
        for j in range(letter_width):
            if letter_matrix[i, j] == 1:
                # Центр текущего пикселя
                center_x = start_x + j * pixel_size + pixel_size // 2
                center_y = start_y + i * pixel_size + pixel_size // 2
                
                # Рисуем пиксель с эффектами
                for px in range(pixel_size):
                    for py in range(pixel_size):
                        x = start_x + j * pixel_size + px
                        y = start_y + i * pixel_size + py
                        if 0 <= x < window_width and 0 <= y < window_height:
                            # Применяем градиент внутри пикселя
                            dist_to_center = distance_to_center(x, y, center_x, center_y)
                            gradient_factor = 1.0 - dist_to_center / (pixel_size * 0.7)
                            gradient_factor = ti.max(0.7, gradient_factor)  # Ограничиваем минимальную яркость
                            
                            # Анимируем цвет
                            animated_color = animate_color(base_color, x, y)
                            
                            # Применяем свечение
                            final_color = apply_bloom(x, y, center_x, center_y, 
                                                    animated_color * gradient_factor, 2.0)
                            
                            pixels[x, y] = final_color

# Функция для фонового свечения
@ti.kernel
def draw_background_glow():
    # Создаем эффект пульсирующего свечения вокруг букв
    t = time[None]
    for i, j in pixels:
        # Добавляем небольшой световой эффект
        glow = 0.02 * ti.sin(0.1 * i + 0.1 * j + t * 2.0) + 0.02
        pixels[i, j] += ti.Vector([glow, glow, glow])

# Функция для эффекта постобработки
@ti.kernel
def post_process():
    for i, j in pixels:
        # Насыщенность цветов
        color = pixels[i, j]
        lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        saturation = 1.2  # Увеличиваем насыщенность
        pixels[i, j] = ti.Vector([
            lum + saturation * (color[0] - lum),
            lum + saturation * (color[1] - lum),
            lum + saturation * (color[2] - lum)
        ])
        
        # Виньетирование (затемнение краев)
        center_x = window_width / 2
        center_y = window_height / 2
        dx = (i - center_x) / window_width
        dy = (j - center_y) / window_height
        dist = ti.sqrt(dx * dx + dy * dy)
        vignette = 1.0 - dist * 0.5
        pixels[i, j] *= vignette

def main():
    # Выводим информацию о запуске
    print("Запуск приложения 'АГПК из пикселей'")
    print(f"Taichi версия: {ti.__version__}")
    
    # Создаем окно
    window = ti.ui.Window("АГПК из пикселей с шейдерами", (window_width, window_height))
    canvas = window.get_canvas()
    
    # Преобразуем матрицы букв в numpy массивы
    np_letter_A = np.array(letter_A)
    np_letter_G = np.array(letter_G)
    np_letter_P = np.array(letter_P)
    np_letter_K = np.array(letter_K)
    
    # Размеры для расположения букв
    letter_width = 4  # Ширина каждой буквы в пикселях
    letter_height = 5  # Высота каждой буквы в пикселях
    spacing = 7  # Расстояние между буквами в пикселях (увеличено для теней)
    
    # Базовые цвета для букв
    color_A = ti.Vector([1.0, 0.2, 0.2])  # Красный
    color_G = ti.Vector([0.2, 1.0, 0.2])  # Зеленый
    color_P = ti.Vector([0.2, 0.2, 1.0])  # Синий
    color_K = ti.Vector([1.0, 1.0, 0.2])  # Желтый
    
    # Вычисляем начальные координаты для центрирования букв
    total_width = 4 * letter_width * pixel_size + 3 * spacing * pixel_size
    start_x = (window_width - total_width) // 2
    start_y = (window_height - letter_height * pixel_size) // 2
    
    # Размер теней
    shadow_offset = 4
    
    # Начальное время
    current_time = 0.0
    
    print("Запуск основного цикла")
    try:
        while window.running:
            # Обновляем время для анимации
            time[None] = current_time
            current_time += 0.02
            
            # Очищаем экран
            clear_screen()
            
            # Рисуем буквы с шейдерными эффектами
            draw_letter_with_effects(np_letter_A, start_x, start_y, color_A, shadow_offset)
            draw_letter_with_effects(np_letter_G, start_x + (letter_width + spacing) * pixel_size, 
                                start_y, color_G, shadow_offset)
            draw_letter_with_effects(np_letter_P, start_x + 2 * (letter_width + spacing) * pixel_size, 
                                start_y, color_P, shadow_offset)
            draw_letter_with_effects(np_letter_K, start_x + 3 * (letter_width + spacing) * pixel_size, 
                                start_y, color_K, shadow_offset)
            
            # Добавляем фоновое свечение
            draw_background_glow()
            
            # Применяем эффекты постобработки
            post_process()
            
            # Отображаем результат
            canvas.set_image(pixels)
            window.show()
    except Exception as e:
        print(f"Произошла ошибка в основном цикле: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        
        # Чтобы окно с ошибкой не закрывалось сразу в случае запуска из EXE
        if getattr(sys, 'frozen', False):
            input("Нажмите Enter для выхода...") 