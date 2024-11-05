import numpy as np
import multiprocessing
import time
import itertools
import math
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from scipy.ndimage import zoom, sobel, gaussian_filter
from scipy.spatial import ConvexHull, Voronoi
from scipy.ndimage import median_filter

SC = 0.0125 
USE_SMOOTH_NOISE = 0
TEX_WIDTH = 64
TEX_HEIGHT = 64

# generate a random matrix based on a seed
SEED = np.random.randint(0, 99999999)
np.random.seed(SEED) 

starting_matrix = np.random.uniform(-1, 1, (TEX_WIDTH, TEX_HEIGHT))

def fract(x):
    return x - np.floor(x)

# value noise and its derivatives
def noised(x, noise_array):
    f = fract(x)

    if USE_SMOOTH_NOISE == 0:
        u = f * f * (3.0 - 2.0 * f)
        du = 6.0 * f * (1.0 - f)
    else:
        u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0)
        du = 30.0 * f * f * (f * (f - 2.0) + 1.0)

    p = np.floor(x).astype(int) % TEX_WIDTH

    a = noise_array[p[0], p[1]]
    b = noise_array[(p[0] + 1) % TEX_WIDTH, p[1]]
    c = noise_array[p[0], (p[1] + 1) % TEX_WIDTH]
    d = noise_array[(p[0] + 1) % TEX_WIDTH, (p[1] + 1) % TEX_WIDTH]

    result_x = a + (b - a) * u[0] + (c - a) * u[1] + (a - b - c + d) * u[0] * u[1]
    result_yz = du * np.array([b - a, c - a]) + (a - b - c + d) * u[::-1]

    return np.array([result_x, result_yz[0], result_yz[1]])

# rotation matrix (similar to mat2 in GLSL)
m2 = np.array([[0.8, -0.6],
               [0.6,  0.8]])

# terrain height map modifier function
def terrainH(x, noise_array):
    p = x * 0.003 / SC
    a = 0.0
    b = 1.0
    d = np.array([0.0, 0.0])

    for _ in range(24):
        n = noised(p, noise_array)
        d += n[1:]
        a += b * n[0] / (1.0 + np.dot(d, d))
        b *= 0.5
        p = np.dot(m2, p) * 2.0

    if USE_SMOOTH_NOISE == 1:
        a *= 0.9

    return SC * 120.0 * a

# terrain height map modifier function
def terrainM(x, noise_array):
    p = x * 0.003 / SC
    a = 0.0
    b = 1.0
    d = np.array([0.0, 0.0])

    for _ in range(9):  # more octaves -> takes longer but smoothens out more
        n = noised(p, noise_array)
        d += n[1:]
        a += b * n[0] / (1.0 + np.dot(d, d))
        b *= 0.5
        p = np.dot(m2, p) * 2.0

    if USE_SMOOTH_NOISE == 1:
        a *= 0.9

    return SC * 120.0 * a

# calculate normal at a given position
def calcNormal(pos, t, noise_array):
    eps = np.array([0.001 * t, 0.0])
    return np.linalg.norm([
        terrainH(pos[:2] - eps, noise_array) - terrainH(pos[:2] + eps, noise_array),
        2.0 * eps[0],
        terrainH(pos[:2] - eps[::-1], noise_array) - terrainH(pos[:2] + eps[::-1], noise_array)
    ])

# fractional brownian motion (fbm) function
def fbm(p, noise_array):
    f = 0.0
    f += 0.5000 * noise_array[int(p[0] % TEX_WIDTH), int(p[1] % TEX_HEIGHT)]
    p = np.dot(m2, p) * 2.02
    f += 0.2500 * noise_array[int(p[0] % TEX_WIDTH), int(p[1] % TEX_HEIGHT)]
    p = np.dot(m2, p) * 2.03
    f += 0.1250 * noise_array[int(p[0] % TEX_WIDTH), int(p[1] % TEX_HEIGHT)]
    p = np.dot(m2, p) * 2.01
    f += 0.0625 * noise_array[int(p[0] % TEX_WIDTH), int(p[1] % TEX_HEIGHT)]

    return f / 0.9375

def generate_terrain():
    terrain_map = np.zeros((TEX_WIDTH, TEX_HEIGHT))

    for i in range(TEX_WIDTH):
        for j in range(TEX_HEIGHT):
            # position vector in the texture grid
            pos = np.array([i, j], dtype=float)
            height_value = terrainH(pos, starting_matrix)
            terrain_map[i, j] = height_value

    return terrain_map

def calculate_drainage(matrix):
    TEX_WIDTH, TEX_HEIGHT = matrix.shape
    drainage_map = np.zeros_like(matrix)

    # create mask for tiles above sea level
    sea_level = 0
    above_sea_level = matrix > sea_level

    y, x = np.meshgrid(np.arange(TEX_HEIGHT), np.arange(TEX_WIDTH))
    flat_height = matrix[above_sea_level]
    flat_x = x[above_sea_level]
    flat_y = y[above_sea_level]

    # sort indices from highest to lowest
    sort_indices = np.argsort(flat_height)[::-1]
    flat_height = flat_height[sort_indices]
    flat_x = flat_x[sort_indices]
    flat_y = flat_y[sort_indices]

    # neighbor offsets
    neighbor_offsets = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])

    # process from highest point to calculate drainage
    for height, x, y in zip(flat_height, flat_x, flat_y):
        neighbors = np.array([x, y]) + neighbor_offsets
        valid_neighbors = (neighbors[:, 0] >= 0) & (neighbors[:, 0] < TEX_WIDTH) & \
                          (neighbors[:, 1] >= 0) & (neighbors[:, 1] < TEX_HEIGHT)

        valid_x = neighbors[valid_neighbors, 0]
        valid_y = neighbors[valid_neighbors, 1]

        downhill = matrix[valid_x, valid_y] < height
        drainage_map[valid_x[downhill], valid_y[downhill]] += 0.25

    return drainage_map

def extend_matrix(matrix, width, height):
    extended_matrix = np.zeros((width + 1, height + 1))
    extended_matrix[:-1, :-1] = matrix
    extended_matrix[-1, :-1] = matrix[-1, :]
    extended_matrix[:-1, -1] = matrix[:, -1]
    extended_matrix[-1, -1] = matrix[-1, -1] 
    return extended_matrix


def reduce_matrix(matrix):
    # reduce the matrix back to 2^n 
    reduced_matrix = matrix[:-1, :-1]
    return reduced_matrix

def midpoint_displacement(matrix, roughness):
    # matrix is of size 2^n + 1, e.g., 129x129, 257x257, 513x513, ...
    size = matrix.shape[0] - 1
    assert (size & (size - 1) == 0) and size != 0, "Matrix size must be 2^n + 1"

    matrix[0, 0] = np.random.uniform(-1, 1)
    matrix[0, size] = np.random.uniform(-1, 1)
    matrix[size, 0] = np.random.uniform(-1, 1)
    matrix[size, size] = np.random.uniform(-1, 1)

    step_size = size
    displacement = roughness

    while step_size > 1:
        half_step = step_size // 2

        # square step
        for x in range(0, size, step_size):
            for y in range(0, size, step_size):
                mid_x = x + half_step
                mid_y = y + half_step

                if mid_x < size and mid_y < size:
                    avg = (matrix[x, y] + matrix[x + step_size, y] +
                           matrix[x, y + step_size] + matrix[x + step_size, y + step_size]) / 4
                    matrix[mid_x, mid_y] = avg + np.random.uniform(-displacement, displacement)

        # diamond step
        for x in range(0, size + 1, half_step):
            for y in range((x + half_step) % step_size, size + 1, step_size):
                sum_vals = 0
                count = 0

                if x >= half_step:
                    sum_vals += matrix[x - half_step, y]
                    count += 1
                if x + half_step <= size:
                    sum_vals += matrix[x + half_step, y]
                    count += 1
                if y >= half_step:
                    sum_vals += matrix[x, y - half_step]
                    count += 1
                if y + half_step <= size:
                    sum_vals += matrix[x, y + half_step]
                    count += 1

                avg = sum_vals / count
                matrix[x, y] = avg + np.random.uniform(-displacement, displacement)

        displacement *= 2 ** (-roughness)
        step_size //= 2

    # normalize the result to [-1, 1]
    matrix = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix)) * 2 - 1

    return matrix

def normalize(matrix):
    min_val = np.min(matrix)
    max_val = np.max(matrix)

    # min-max scaling [-1, 1] -- matrix - min_val for [0, 1]
    matrix = (matrix - min_val) / (max_val - min_val)

    return matrix

def upscale(matrix, scale_factor):
    return np.kron(matrix, np.ones((scale_factor, scale_factor)))

def gaussian_blur(matrix, sigma=1.0):
    return gaussian_filter(matrix, sigma=sigma)

def visualize_3d(noise_3d):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = np.arange(0, noise_3d.shape[1], 1)
    y = np.arange(0, noise_3d.shape[0], 1)
    x, y = np.meshgrid(x, y)

    surf = ax.plot_surface(x, y, noise_3d, cmap='terrain', linewidth=0, antialiased=False)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z (Depth)')
    ax.set_title('3D Noise Visualization')

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

terrain_map = (generate_terrain())
midpoint_matrix = ((reduce_matrix(midpoint_displacement(extend_matrix(terrain_map, TEX_WIDTH, TEX_HEIGHT), roughness=1.125))))

drainage_map = calculate_drainage(midpoint_matrix)
combined_map = ((drainage_map)) * (midpoint_matrix)

plt.imshow(gaussian_blur(combined_map), cmap='terrain') 
plt.colorbar(label='Height')
plt.title('Smoothed 2D Terrain Map')
plt.xlabel('Width')
plt.ylabel('Height')
plt.show()

visualize_3d(combined_map)
