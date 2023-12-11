# Rescale matrix between 0 and 128
# Place it in the middle of the matrix 1920x1080 which are initialised to 0.


import numpy as np
import matplotlib.pyplot as plt

# Function to rescale a matrix between a new min and max
def rescale_matrix(matrix, new_min, new_max):
    old_min, old_max = matrix.min(), matrix.max()
    return (new_max - new_min) / (old_max - old_min) * (matrix - old_min) + new_min

# Load the matrix
filepath = 'slm_matrix_gen/test_phase_pattern_150mp_3gf.npy'
matrix = np.load(filepath)

# Rescale matrix between 0 and 128
rescaled_matrix = rescale_matrix(matrix, 0, 128)

# Create a 1920x1080 matrix initialized to 0
large_matrix = np.zeros((1080, 1920))

# Calculate center coordinates
center_x = (large_matrix.shape[1] - rescaled_matrix.shape[1]) // 2
center_y = (large_matrix.shape[0] - rescaled_matrix.shape[0]) // 2

# Place the rescaled matrix in the center of the larger matrix
large_matrix[center_y:center_y + rescaled_matrix.shape[0], center_x:center_x + rescaled_matrix.shape[1]] = rescaled_matrix

# Optionally, display the matrix using matplotlib
plt.imshow(large_matrix, cmap='gray')
#plt.show()

# Save the array to a CSV file
np.savetxt('slm_matrix_gen/test_phase_pattern_150mp_3gf_slm_ready.csv', large_matrix, delimiter=',')


