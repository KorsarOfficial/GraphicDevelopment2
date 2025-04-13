import taichi as ti
import numpy as np
import math

ti.init(arch=ti.cpu)

window_width, window_height = 800, 600
pixel_size = 20

pixels = ti.Vector.field(3, dtype=ti.f32, shape=(window_width, window_height))

time = ti.field(dtype=ti.f32, shape=())

letter_A = [
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [1, 1, 1, 1],
    [1, 0, 0, 1],
    [1, 0, 0, 1]
]

letter_G = [
    [1, 1, 1, 1],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 1, 1, 1]
]

letter_P = [
    [1, 1, 1, 1],
    [1, 0, 0, 1],
    [1, 1, 1, 1],
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

@ti.kernel
def clear_screen():
    for i, j in pixels:
        bg_color = ti.Vector([0.05, 0.05, 0.1]) + ti.Vector([0.0, 0.0, 0.05]) * ti.sin(0.01 * i + time[None])
        pixels[i, j] = bg_color

@ti.func
def distance_to_center(x: ti.i32, y: ti.i32, center_x: ti.i32, center_y: ti.i32) -> ti.f32:
    dx = x - center_x
    dy = y - center_y
    return ti.sqrt(dx * dx + dy * dy)

@ti.func
def apply_bloom(x: ti.i32, y: ti.i32, center_x: ti.i32, center_y: ti.i32, 
                base_color: ti.types.vector(3, ti.f32), intensity: ti.f32) -> ti.types.vector(3, ti.f32):
    dist = distance_to_center(x, y, center_x, center_y)
    bloom_factor = ti.exp(-dist * 0.01 * intensity)
    return base_color * (1.0 + bloom_factor * 0.5)

@ti.func
def draw_shadow(x: ti.i32, y: ti.i32, shadow_offset_x: ti.i32, shadow_offset_y: ti.i32):
    shadow_x = x + shadow_offset_x
    shadow_y = y + shadow_offset_y
    
    if 0 <= shadow_x < window_width and 0 <= shadow_y < window_height:
        shadow_color = ti.Vector([0.0, 0.0, 0.0])
        current_color = pixels[shadow_x, shadow_y]
        shadow_strength = 0.7
        pixels[shadow_x, shadow_y] = current_color * (1 - shadow_strength) + shadow_color * shadow_strength

@ti.func
def animate_color(base_color: ti.types.vector(3, ti.f32), x: ti.i32, y: ti.i32) -> ti.types.vector(3, ti.f32):
    pulse = 0.2 * ti.sin(time[None] * 2.0 + 0.1 * x + 0.1 * y) + 1.0
    rainbow_effect = ti.Vector([
        0.5 * ti.sin(time[None] + 0.1 * x) + 0.5,
        0.5 * ti.sin(time[None] + 0.1 * y + 2.0) + 0.5,
        0.5 * ti.sin(time[None] + 0.1 * (x + y) + 4.0) + 0.5
    ])
    mix_factor = 0.3
    result_color = base_color * (1.0 - mix_factor) + rainbow_effect * mix_factor
    return result_color * pulse

@ti.kernel
def draw_letter_with_effects(letter_matrix: ti.types.ndarray(), start_x: ti.i32, start_y: ti.i32, 
                           base_color: ti.types.vector(3, ti.f32), shadow_offset: ti.i32):
    letter_height = letter_matrix.shape[0]
    letter_width = letter_matrix.shape[1]
    
    for i in range(letter_height):
        for j in range(letter_width):
            if letter_matrix[i, j] == 1:
                center_x = start_x + j * pixel_size + pixel_size // 2
                center_y = start_y + i * pixel_size + pixel_size // 2
                
                for px in range(pixel_size):
                    for py in range(pixel_size):
                        x = start_x + j * pixel_size + px
                        y = start_y + i * pixel_size + py
                        if 0 <= x < window_width and 0 <= y < window_height:
                            draw_shadow(x, y, shadow_offset, shadow_offset)
    
    for i in range(letter_height):
        for j in range(letter_width):
            if letter_matrix[i, j] == 1:
                center_x = start_x + j * pixel_size + pixel_size // 2
                center_y = start_y + i * pixel_size + pixel_size // 2
                
                for px in range(pixel_size):
                    for py in range(pixel_size):
                        x = start_x + j * pixel_size + px
                        y = start_y + i * pixel_size + py
                        if 0 <= x < window_width and 0 <= y < window_height:
                            dist_to_center = distance_to_center(x, y, center_x, center_y)
                            gradient_factor = 1.0 - dist_to_center / (pixel_size * 0.7)
                            gradient_factor = ti.max(0.7, gradient_factor)
                            
                            animated_color = animate_color(base_color, x, y)
                            
                            final_color = apply_bloom(x, y, center_x, center_y, 
                                                    animated_color * gradient_factor, 2.0)
                            
                            pixels[x, y] = final_color

@ti.kernel
def draw_background_glow():
    t = time[None]
    for i, j in pixels:
        glow = 0.02 * ti.sin(0.1 * i + 0.1 * j + t * 2.0) + 0.02
        pixels[i, j] += ti.Vector([glow, glow, glow])

@ti.kernel
def post_process():
    for i, j in pixels:
        color = pixels[i, j]
        lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        saturation = 1.2
        pixels[i, j] = ti.Vector([
            lum + saturation * (color[0] - lum),
            lum + saturation * (color[1] - lum),
            lum + saturation * (color[2] - lum)
        ])
        
        center_x = window_width / 2
        center_y = window_height / 2
        dx = (i - center_x) / window_width
        dy = (j - center_y) / window_height
        dist = ti.sqrt(dx * dx + dy * dy)
        vignette = 1.0 - dist * 0.5
        pixels[i, j] *= vignette

def main():
    window = ti.ui.Window("АГПК из пикселей с шейдерами", (window_width, window_height))
    canvas = window.get_canvas()
    
    np_letter_A = np.array(letter_A)
    np_letter_G = np.array(letter_G)
    np_letter_P = np.array(letter_P)
    np_letter_K = np.array(letter_K)
    
    letter_width = 4
    letter_height = 5
    spacing = 7
    
    color_A = ti.Vector([1.0, 0.2, 0.2])
    color_G = ti.Vector([0.2, 1.0, 0.2])
    color_P = ti.Vector([0.2, 0.2, 1.0])
    color_K = ti.Vector([1.0, 1.0, 0.2])
    
    total_width = 4 * letter_width * pixel_size + 3 * spacing * pixel_size
    start_x = (window_width - total_width) // 2
    start_y = (window_height - letter_height * pixel_size) // 2
    
    shadow_offset = 4
    
    current_time = 0.0
    
    while window.running:
        time[None] = current_time
        current_time += 0.02
        
        clear_screen()
        
        draw_letter_with_effects(np_letter_A, start_x, start_y, color_A, shadow_offset)
        draw_letter_with_effects(np_letter_G, start_x + (letter_width + spacing) * pixel_size, 
                               start_y, color_G, shadow_offset)
        draw_letter_with_effects(np_letter_P, start_x + 2 * (letter_width + spacing) * pixel_size, 
                               start_y, color_P, shadow_offset)
        draw_letter_with_effects(np_letter_K, start_x + 3 * (letter_width + spacing) * pixel_size, 
                               start_y, color_K, shadow_offset)
        
        draw_background_glow()
        
        post_process()
        
        canvas.set_image(pixels)
        window.show()

if __name__ == "__main__":
    main() 