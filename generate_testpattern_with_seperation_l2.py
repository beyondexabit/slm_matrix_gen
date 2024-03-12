import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import sawtooth
import sys
from numpy.fft import fft2, fftshift, ifft2 # Python DFT

# Function to rescale a matrix between a new min and max
def rescale_matrix(matrix, new_min, new_max):
    old_min, old_max = matrix.min(), matrix.max()
    return (new_max - new_min) / (old_max - old_min) * (matrix - old_min) + new_min

def generate_phase_pattern(matrix, macropixel_size, macropixel_seperation, grating_frequency_x, grating_frequency_y):
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

    elements = input_matrix[0]

    nrows, ncols = matrix.shape
    pattern = np.zeros((nrows * macropixel_size, ncols * macropixel_size + (ncols - 1) * macropixel_seperation))
    freq_shift_x = 20
    freq_shift_y = 36

    count = 0
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

            pattern[i * macropixel_size:(i + 1) * macropixel_size, 
                    j * (macropixel_size + j*macropixel_seperation):(j + 1) * (macropixel_size + j*macropixel_seperation)] = grating_combined

            grating_frequency_x += freq_shift_x

        grating_frequency_y += freq_shift_y
        grating_frequency_x = grating_freq_x_old
        count += 1

    # add this for correct scaling  ----  SEEMS TO NOT WORK FOR DIFFRACTSIM
    #pattern[0,0] = np.pi*2
    #pattern[np.shape(pattern)[0] - 1, 0] = 0
        
    return pattern


# for layer 2:
# 2* 1.3mm = 2.6mm 
# height of slm : 8.64mm 
# 8.64 /3 = 2.88mm / 2 = 1.44mm
# 1080/3 = 360pixels
# need 0.2 mm seperation
# 15360 = 1920
# 2000 = 250pixel seperation

# Example usage

input_matrix = np.array([[1, 0.1], [0.8, 0.3], [0, 0.6]])
macropixel_size = 360
grating_frequency_y = -36  # Frequency offset for y-axis

grating_frequency_x = 20

macropixel_seperation = 250
matrix_new = generate_phase_pattern(input_matrix, macropixel_size, macropixel_seperation, grating_frequency_x, grating_frequency_y)

# Rescale matrix between 0 and 128
rescaled_matrix = rescale_matrix(matrix_new, 0, 254)

# Plot the phase pattern
#plt.imshow(rescaled_matrix, cmap='gray')
#plt.figure(figsize=(10, 10))
#plt.show()

# Create a 1920x1080 matrix initialized to 0
#large_matrix = np.zeros((1080, 1920))
large_matrix = np.zeros((1080, 1920))

# Calculate center coordinates
center_x = (large_matrix.shape[1] - rescaled_matrix.shape[1]) // 2
center_y = (large_matrix.shape[0] - rescaled_matrix.shape[0]) // 2

# Place the rescaled matrix in the center of the larger matrix
large_matrix[center_y:center_y + rescaled_matrix.shape[0], center_x:center_x + rescaled_matrix.shape[1]] = rescaled_matrix

plt.figure()
plt.imshow(large_matrix, cmap='gray')


F = fft2(large_matrix)                         
F = fftshift(F)
P = np.abs(F)  

# Apply logarithmic scaling to the image data
log_scaled_image = np.log(1 + P)

plt.figure()
plt.imshow(log_scaled_image, cmap='viridis', interpolation='nearest')
plt.title('Magnitude of Fourier Transform')


#plt.show()


np.save(f'/Users/jakubkostial/Documents/phd/code/slm_matrix_gen-main/matrices_dsim_scaling/l2' +
        f'/phase_pattern_l2_{grating_frequency_x}gfx{grating_frequency_y}gfy_1_0.1_0.8_0.3_0_0.6.npy', large_matrix)
