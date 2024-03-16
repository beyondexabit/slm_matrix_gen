import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft2, fftshift, ifft2 # Python DFT

matrix = np.load('/Users/jakubkostial/Documents/phd/code/slm_matrix_gen-main/repo/slm_matrix_gen/formatted_matrices_individual/l1/phase_pattern_l1_20gfx-46gfy_00_10_00_test.npy')

plt.figure()
plt.imshow(matrix, cmap='viridis', interpolation='nearest')
plt.title('SLM visualisation')


F = fft2(matrix)                         
F = fftshift(F)
P = np.abs(F)  

# Apply logarithmic scaling to the image data
log_scaled_image = np.log(1 + P)

plt.figure()
plt.imshow(log_scaled_image, cmap='viridis', interpolation='nearest')
plt.title('Magnitude of Fourier Transform')
plt.show()
