import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft2, fftshift, ifft2 # Python DFT

matrix = np.load('/Users/jakubkostial/Documents/phd/code/loop_matmul/repo/formatted_matrices/saw_test_phase_pattern_150mp_20gf.npy')

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
