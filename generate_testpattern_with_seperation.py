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
    freq_shift_y = 46

    count = 0
    for i in range(nrows):
        for j in range(ncols):
            # Determine the amplitude of the grating based on the matrix value
            amplitude = np.abs(matrix[i, j])
            phase_shift = np.pi if matrix[i, j] < 0 else 0

            # Create sinusoidal grating patterns for the macropixel
            x = np.linspace(0, 2 * np.pi * grating_frequency_x, macropixel_size) # grating frequency parameter determines how many times it occurs in the range
            y = np.linspace(0, 2 * np.pi * grating_frequency_y, macropixel_size) # aka oscillations per macropixel

            grating_x = np.pi/2 * (sawtooth(x) + 1) 
            grating_y = np.pi/2 * (sawtooth(y) + 1) 

            # Combine the x and y gratings with modulo operation to wrap around π
            grating_combined = (np.outer(grating_y, np.ones_like(grating_x)) + np.outer(np.ones_like(grating_y), grating_x)) % np.pi
            grating_combined *= amplitude
            grating_combined += phase_shift

            pattern[i * macropixel_size:(i + 1) * macropixel_size, 
                    j * (macropixel_size + j*macropixel_seperation):(j + 1) * (macropixel_size + j*macropixel_seperation)] = grating_combined

            grating_frequency_x += freq_shift_x

            # Calculate center coordinates of the placed grating_combined within pattern
            center_x_grating_combined = (np.shape(grating_combined)[0] - 1) // 2
            center_y_grating_combined = (np.shape(grating_combined)[1] - 1) // 2

            # Print or use the center_element as needed
            center_x = i * macropixel_size + center_x_grating_combined
            center_y = j * (macropixel_size + j*macropixel_seperation) + center_y_grating_combined
            print(f"Center element of grating_combined:{center_x},{center_y}")


        grating_frequency_y += freq_shift_y
        grating_frequency_x = grating_freq_x_old
        count += 1

    # add this for correct scaling     
    #pattern[0,0] = np.pi*2
    #pattern[0,1] = 0


    return pattern


# Example usage
input_matrix = np.array([[0, 0], [0, 0], [0, 1]])
macropixel_size = 216
grating_frequency_y = -46  # Frequency offset for y-axis

grating_frequency_x = 20

macropixel_seperation = 108
matrix_new = generate_phase_pattern(input_matrix, macropixel_size, macropixel_seperation, grating_frequency_x, grating_frequency_y)

# Rescale matrix between 0 and 128
rescaled_matrix = rescale_matrix(matrix_new, 0, 255)

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

#large_matrix = np.rot90(large_matrix, k=2)    
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

np.save(f'/Users/jakubkostial/Documents/phd/code/slm_matrix_gen-main/repo/slm_matrix_gen/formatted_matrices_individual/l1' +
        f'/phase_pattern_l1_{grating_frequency_x}gfx{grating_frequency_y}gfy_00_00_01_test.npy', large_matrix)

plt.show()

#                                   y,  x
#Center element of grating_combined:324,798
#Center element of grating_combined:324,1122
#Center element of grating_combined:540,798
#Center element of grating_combined:540,1122
#Center element of grating_combined:751,798
#Center element of grating_combined:751,1122

