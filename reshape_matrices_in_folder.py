# Rescale matrix between 0 and 128
# Place it in the middle of the matrix 1920x1080 which are initialised to 0.
import numpy as np
import matplotlib.pyplot as plt
import os

# Function to rescale a matrix between a new min and max
def rescale_matrix(matrix, new_min, new_max):
    old_min, old_max = matrix.min(), matrix.max()
    return (new_max - new_min) / (old_max - old_min) * (matrix - old_min) + new_min

# Specify the directory you want to read from
folder_path = '/Users/jakubkostial/Documents/phd/code/loop_matmul/repo/matrices/'

# Iterate over all files in the directory
for filename in os.listdir(folder_path):
    #file_path = os.path.join(folder_path, filename)
    # Check if it's a file, not a directory
    # Open and read the file
    filepath = '/Users/jakubkostial/Documents/phd/code/loop_matmul/repo/matrices/'
    matrix = np.load(filepath+filename)
    # Rescale matrix between 0 and 128
    rescaled_matrix = rescale_matrix(matrix, 0, 256)
    # Create a 1920x1080 matrix initialized to 0
    large_matrix = np.zeros((1080, 1920))
    # Calculate center coordinates
    center_x = (large_matrix.shape[1] - rescaled_matrix.shape[1]) // 2
    center_y = (large_matrix.shape[0] - rescaled_matrix.shape[0]) // 2
    # Place the rescaled matrix in the center of the larger matrix
    large_matrix[center_y:center_y + rescaled_matrix.shape[0], center_x:center_x + rescaled_matrix.shape[1]] = rescaled_matrix
    # Save the array to a CSV file
    np.savetxt(f'/Users/jakubkostial/Documents/phd/code/loop_matmul/repo/formatted_matrices/slm_ready_{filename[:-4]}.csv', large_matrix, delimiter=',')
