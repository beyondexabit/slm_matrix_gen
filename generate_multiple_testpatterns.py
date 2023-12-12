import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import sawtooth

# Function to rescale a matrix between a new min and max
def rescale_matrix(matrix, new_min, new_max):
    old_min, old_max = matrix.min(), matrix.max()
    return (new_max - new_min) / (old_max - old_min) * (matrix - old_min) + new_min

def generate_phase_pattern(matrix, macropixel_size, grating_frequency_x, grating_frequency_y):
    """
    Generate a phase pattern for an LC-SLM based on the given matrix.
    
    Args:
    - matrix (2D numpy array): The matrix to be encoded, with values from -1 to 1.
    - macropixel_size (int): The size of each macropixel (n x n pixels).
    - grating_frequency_x (float): Frequency of the grating pattern within each macropixel on the x-axis.
    - grating_frequency_y (float): Frequency of the grating pattern within each macropixel on the y-axis.
    
    Returns:
    - 2D numpy array: The generated phase pattern.
    """
    grating_freq_x_old = grating_frequency_x 

    nrows, ncols = matrix.shape
    pattern = np.zeros((nrows * macropixel_size, ncols * macropixel_size))
    freq_shift_x = 4
    freq_shift_y = 4

    for i in range(nrows):
        for j in range(ncols):
            # Determine the amplitude of the grating based on the matrix value

            amplitude = np.abs(matrix[i, j])
            phase_shift = np.pi if matrix[i, j] < 0 else 0

            # Create sinusoidal grating patterns for the macropixel
            x = np.linspace(0, 2 * np.pi * grating_frequency_x, macropixel_size)
            y = np.linspace(0, 2 * np.pi * grating_frequency_y, macropixel_size)

            grating_x = np.pi/2 * (sawtooth(x) + 1) 
            grating_y = np.pi/2 * (sawtooth(y) + 1) 

            # Combine the x and y gratings with modulo operation to wrap around π
            grating_combined = (np.outer(grating_y, np.ones_like(grating_x)) + np.outer(np.ones_like(grating_y), grating_x)) % np.pi
            grating_combined *= amplitude
            grating_combined += phase_shift

            # Place the macropixel in the pattern
            pattern[i * macropixel_size:(i + 1) * macropixel_size, j * macropixel_size:(j + 1) * macropixel_size] = grating_combined

            grating_frequency_x += freq_shift_x

        grating_frequency_y += freq_shift_y
        grating_frequency_x = grating_freq_x_old

    return pattern


# Example usage
input_matrix = np.array([[1, 0, -1], [0, -1, 0], [0, 0, 1]])
macropixel_size = 150
grating_frequency_y = 0  # Frequency offset for y-axis


for i in range(1, 11):
    print(i)
    grating_frequency_x = i*10
    matrix_new = generate_phase_pattern(input_matrix, macropixel_size, grating_frequency_x, grating_frequency_y)

    # Rescale matrix between 0 and 128
    rescaled_matrix = rescale_matrix(matrix_new, 0, 256)

    # Create a 1920x1080 matrix initialized to 0
    large_matrix = np.zeros((1080, 1920))

    # Calculate center coordinates
    center_x = (large_matrix.shape[1] - rescaled_matrix.shape[1]) // 2
    center_y = (large_matrix.shape[0] - rescaled_matrix.shape[0]) // 2

    # Place the rescaled matrix in the center of the larger matrix
    large_matrix[center_y:center_y + rescaled_matrix.shape[0], center_x:center_x + rescaled_matrix.shape[1]] = rescaled_matrix

    np.save(f'/Users/jakubkostial/Documents/phd/code/loop_matmul/repo/formatted_matrices/saw_test_phase_pattern_150mp_{grating_frequency_x}gf.npy', large_matrix)