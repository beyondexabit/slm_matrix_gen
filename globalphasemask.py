import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift, ifft2 # Python DFT
from scipy.signal import sawtooth

# Function to rescale a matrix between a new min and max
def rescale_matrix(matrix, new_min, new_max):
    old_min, old_max = matrix.min(), matrix.max()
    return (new_max - new_min) / (old_max - old_min) * (matrix - old_min) + new_min

grating_frequency_x = 460
grating_frequency_y = 0

matrix = np.zeros((1080, 1920))

x = np.linspace(0, 2 * np.pi * grating_frequency_x, matrix.shape[1])
y = np.linspace(0, 2 * np.pi * grating_frequency_y, matrix.shape[0])

grating_x = np.pi/2 * (sawtooth(x) + 1) 
grating_y = np.pi/2 * (sawtooth(y) + 1) 

amplitude = 1   
phase_shift = 0

# Combine the x and y gratings with modulo operation to wrap around π
grating_combined = (np.outer(grating_y, np.ones_like(grating_x)) + np.outer(np.ones_like(grating_y), grating_x)) % np.pi
grating_combined *= amplitude
grating_combined += phase_shift

grating_combined = rescale_matrix(grating_combined, 0, 254)
'''
large_matrix = np.zeros((1080, 1920))

# Calculate center coordinates
center_x = (large_matrix.shape[1] - grating_combined.shape[1]) // 2
center_y = (large_matrix.shape[0] - grating_combined.shape[0]) // 2

center_x = 1024-400
# Place the rescaled matrix in the center of the larger matrix
large_matrix[center_y:center_y + grating_combined.shape[0], center_x:center_x + grating_combined.shape[1]] = grating_combined


'''
plt.figure()
plt.imshow(grating_combined , cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.title('Grating pattern')
#plt.show()

np.save('/Users/jakubkostial/Documents/phd/code/slm_matrix_gen-main/matrices_dsim_scaling/l1/global_mask.npy', grating_combined)    

