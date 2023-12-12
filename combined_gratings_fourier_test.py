import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import sawtooth
from numpy.fft import fft2, fftshift, ifft2 # Python DFT
import sys

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
    grating_freq_y_old = grating_frequency_y 

    nrows, ncols = matrix.shape
    pattern = np.zeros((nrows * macropixel_size, ncols * macropixel_size))
    freq_shift_x = 4
    freq_shift_y = 4

    for i in range(nrows):
        for j in range(ncols):
            # Determine the amplitude of the grating based on the matrix value
            #print(matrix[i, j])
            #print('b')

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

        #print('a')
        grating_frequency_y += freq_shift_y
        grating_frequency_x = grating_freq_x_old

    return pattern


# Example usage
matrix = np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]])
macropixel_size = 150
grating_frequency_x = 10  # Frequency for x-axis
grating_frequency_y = 0  # Frequency for y-axis
phase_pattern = generate_phase_pattern(matrix, macropixel_size, grating_frequency_x, grating_frequency_y)

# Displaying the phase pattern
plt.imshow(phase_pattern, cmap='gray')
plt.colorbar()
#plt.show()

F = fft2(phase_pattern)                         
F = fftshift(F)
P = np.abs(F)  

# Apply logarithmic scaling to the image data
log_scaled_image = np.log(1 + P)

plt.figure()
plt.imshow(log_scaled_image, cmap='viridis', interpolation='nearest')
plt.title('Magnitude of Fourier Transform')
plt.show()

