# Take axb matrix and convert into macropixel matrix 
# where each element is determined by modulating grating amplitude to transmit light to the first order
# to encode negative numbers we give a pi phase shift
# determine grating amplitude and angle


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import sawtooth


def rescaled_sin(sine_values):
    # Rescale the sine values between 0 and 1
    min_val = np.min(sine_values)
    max_val = np.max(sine_values)

    # Perform the rescaling
    rescaled_sinex = (sine_values - min_val) / (max_val - min_val)
    return rescaled_sinex


def generate_phase_pattern(matrix, macropixel_size, grating_frequency):
    """
    Generate a phase pattern for an LC-SLM based on the given matrix.
    
    Args:
    - matrix (2D numpy array): The matrix to be encoded, with values from -1 to 1.
    - macropixel_size (int): The size of each macropixel (n x n pixels).
    - grating_frequency (float): Frequency of the grating pattern within each macropixel.
    
    Returns:
    - 2D numpy array: The generated phase pattern.
    """
    nrows, ncols = matrix.shape
    pattern = np.zeros((nrows * macropixel_size, ncols * macropixel_size))

    for i in range(nrows):
        for j in range(ncols):
            # Determine the amplitude of the grating based on the matrix value
            amplitude = np.pi * np.abs(matrix[i, j])
            phase_shift = np.pi if matrix[i, j] < 0 else 0

            # Create a sinusoidal grating pattern for the macropixel
            x = np.linspace(0, 2 * np.pi * grating_frequency, macropixel_size)

            grating = amplitude/2 * (sawtooth(x)+1) + phase_shift

            # Place the macropixel in the pattern
            pattern[i * macropixel_size:(i + 1) * macropixel_size, j * macropixel_size:(j + 1) * macropixel_size] = grating
    
    return pattern

# Example usage
matrix = np.array([[0.2, 0.5], [0.7, 1.0], [-0.2, -0.5], [-0.7, -1.0]])
macropixel_size = 150
grating_frequency = -10  # Change this value to adjust the frequency
phase_pattern = generate_phase_pattern(matrix, macropixel_size, grating_frequency)

# Plotting
plt.figure()
plt.imshow(matrix, cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.title('Original Matrix')

plt.figure()
plt.imshow(phase_pattern, cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.title('Phase Pattern')

plt.show()

# Save the array to a .npy file
#np.save('test_phase_pattern_150mp_3gf.npy', phase_pattern)
