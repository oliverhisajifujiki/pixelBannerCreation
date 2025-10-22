import numpy as np
from PIL import Image
import math, random

# --- Parameters ---
rows, cols = 12, 50
scale = 20
palette = ["#562717", "#C21717", "#E76219", "#FEA712"]
fps = 30
filename = "sine_wave_paint_fill.gif"

pairs_per_color = 6
extra_pairs = 2       # extra sine pairs before fill
band = 1.4
speed = 0.02
t_end = 1.0

# --- Helpers ---
def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def make_frame(grid):
    img = Image.fromarray(grid.astype(np.uint8), "RGB")
    return img.resize((cols * scale, rows * scale), Image.NEAREST)

def draw_sine_segment(grid, color, A, wavelength, phase, t, x_max):
    for x in range(x_max):
        y_wave = rows / 2 + A * math.sin(2 * math.pi * (x / wavelength) + phase)
        for y in range(rows):
            if abs(y - y_wave) < band:
                grid[y, x] = color

# --- Setup ---
colors = [hex_to_rgb(c) for c in palette]
grid = np.zeros((rows, cols, 3), dtype=np.uint8)
grid[:, :] = colors[0]
frames = []

# --- Animation ---
num_colors = len(colors)

for repeat in range(2):  # go through the full color cycle twice
    for c_idx in range(num_colors):
        target = colors[(c_idx + 1) % num_colors]  # wrap back to first color
        current = colors[c_idx]

        # reset base grid for this color phase
        grid[:, :] = current

        # wave pairs
        for pair_idx in range(pairs_per_color + extra_pairs):
            A1 = random.uniform(rows / 4, rows / 2.3)
            A2 = random.uniform(rows / 4, rows / 2.3)
            λ1 = random.uniform(cols / 2, cols)
            λ2 = random.uniform(cols / 2, cols)
            φ1 = random.uniform(0, 2 * math.pi)
            φ2 = random.uniform(0, 2 * math.pi)

            t = 0
            while t <= t_end:
                x_max = min(cols, int((t / t_end) * cols) + 1)
                new_grid = np.copy(grid)

                draw_sine_segment(new_grid, target, A1, λ1, φ1, t, x_max)
                draw_sine_segment(new_grid, target, A2, λ2, φ2, t, x_max)

                frames.append(make_frame(new_grid))
                diff_mask = np.any(new_grid != grid, axis=2)
                grid[diff_mask] = new_grid[diff_mask]
                t += speed

        # final smart fill sweep (your serpentine version)
        fill_speed = 0.4
        unpainted_mask = np.any(grid != target, axis=2)
        for x in range(cols):
            y_range = range(rows) if x % 2 == 0 else reversed(range(rows))
            for y in y_range:
                if unpainted_mask[y, x]:
                    grid[y, x] = target
                    for _ in range(int(1 / fill_speed)):
                        frames.append(make_frame(grid))

# --- Add one more frame to fully close the cycle ---
frames.append(make_frame(np.full_like(grid, colors[0])))




# --- Save ---
frames[0].save(
    filename,
    save_all=True,
    append_images=frames[1:],
    duration=int(1000 / fps),
    loop=0,
)
print(f"Saved {filename} with {len(frames)} frames.")
