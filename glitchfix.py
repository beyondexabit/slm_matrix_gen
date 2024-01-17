import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import sawtooth
import sys

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
    freq_shift_x = 10
    freq_shift_y = 10

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
            
            plt.plot(grating_x)
            plt.title('1D NumPy Array Plot')
            plt.xlabel('Index')
            plt.ylabel('Value')

            plt.plot(grating_y)
            plt.title('1D NumPy Array Plot')
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.show()
            
            # Combine the x and y gratings with modulo operation to wrap around π
            grating_combined = (np.outer(grating_y, np.ones_like(grating_x)) + np.outer(np.ones_like(grating_y), grating_x)) % np.pi
            grating_combined *= amplitude
            grating_combined += phase_shift

            # Displaying the phase pattern
            #plt.imshow(grating_combined, cmap='gray')
            #plt.colorbar()
            #plt.show()

            # Place the macropixel in the pattern
            pattern[i * macropixel_size:(i + 1) * macropixel_size, j * macropixel_size:(j + 1) * macropixel_size] = grating_combined

            grating_frequency_x += freq_shift_x

        grating_frequency_y += freq_shift_y
        grating_frequency_x = grating_freq_x_old

    return pattern


# Example usage
input_matrix = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) #np.ones((4, 3))
macropixel_size = 70
grating_frequency_y = 1 # Frequency offset for y-axis



# orig grating_frequency_x = 70

grating_frequency_x = 70  # Frequency for x-axis
grating_frequency_y = 0  # Frequency for y-axis
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

# Displaying the phase pattern
plt.imshow(large_matrix, cmap='gray')
plt.colorbar()
plt.show()
