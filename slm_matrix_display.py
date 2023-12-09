# Take axb matrix and convert into macropixel matrix 
# where each element is determined by modulating grating amplitude to transmit light to the first order
# to encode negative numbers we give a pi phase shift
# determine grating amplitude and angle


import matplotlib.pyplot as plt
import numpy as np

def generate_phase_pattern(matrix, macropixel_size):
    """
    Generate a phase pattern for an LC-SLM based on the given matrix,
    with values ranging from 0 to 1 and grating scaled from 0 to π.
    
    Args:
    - matrix (2D numpy array): The matrix to be encoded, with values from 0 to 1.
    - macropixel_size (int): The size of each macropixel (n x n pixels).
    
    Returns:
    - 2D numpy array: The generated phase pattern.
    """
    nrows, ncols = matrix.shape
    pattern = np.zeros((nrows * macropixel_size, ncols * macropixel_size))

    # Find the minimum and maximum values in the matrix
    min_value = np.min(matrix)
    max_value = np.max(matrix)

    # Scale the matrix to the range [-1, 1]
    scaled_matrix = 2 * (matrix - min_value) / (max_value - min_value) - 1


    for i in range(nrows):
        for j in range(ncols):
            # Scale the matrix value to determine the amplitude of the grating
            if (scaled_matrix[i, j] >= 0):
                amplitude = scaled_matrix[i, j] * np.pi
                # Create the grating pattern for the macropixel
                grating = np.linspace(0, amplitude, macropixel_size)


            if (scaled_matrix[i, j] < 0):
                amplitude = (scaled_matrix[i, j]) * np.pi
                # Create the grating pattern for the macropixel
                grating = np.linspace(np.pi, amplitude, macropixel_size)


            macropixel = grating[:, None]

            # Place the macropixel in the pattern
            pattern[i * macropixel_size:(i + 1) * macropixel_size, j * macropixel_size:(j + 1) * macropixel_size] = macropixel
    
    return pattern

# Example usage
matrix = np.array([[0.2, 0.5], [0.7, 1.0], [-0.2, -0.5], [-0.7, -1.0]])
macropixel_size = 100
phase_pattern = generate_phase_pattern(matrix, macropixel_size)


# Compute the 2D Fourier transform of the matrix
fourier_transform = np.fft.fft2(phase_pattern)
# Shift the zero frequency component to the center
fourier_transform_shifted = np.fft.fftshift(fourier_transform)
# Compute the magnitude spectrum
magnitude_spectrum = 20 * np.log(np.abs(fourier_transform_shifted))

plt.figure()
plt.imshow(matrix, cmap='viridis', interpolation='nearest')
plt.colorbar()  # Add a colorbar to the plot
plt.title('Matrix Heatmap')

plt.figure()
plt.imshow(phase_pattern, cmap='viridis', interpolation='nearest')
plt.colorbar()  # Add a colorbar to the plot
plt.title('Matrix Heatmap')

plt.show()



plt.figure()
# Plotting the original matrix
plt.subplot(121), plt.imshow(matrix, cmap='viridis')
plt.title('Original Matrix'), plt.xticks([]), plt.yticks([])
# Plotting the magnitude spectrum of the Fourier transform
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='viridis')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])



