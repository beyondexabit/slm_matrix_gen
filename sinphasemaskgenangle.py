import numpy as np
import matplotlib.pyplot as plt 
import sys
from scipy.signal import sawtooth


# Function to rescale a matrix between a new min and max
def rescale_matrix(matrix, new_min, new_max):
    old_min, old_max = matrix.min(), matrix.max()
    return (new_max - new_min) / (old_max - old_min) * (matrix - old_min) + new_min


# Define wavelength and wavenumber
wavelength = 1550e-9
k = 2 * np.pi / wavelength

# Create meshgrid coordinates
x = np.arange(-5, 5.01, 0.01)
y = np.arange(-5, 5.01, 0.01)
x, y = np.meshgrid(x, y)  # Combine into meshgrid

# Define aperture radius and create aperture mask
aperture_radius = 0.1
aperture = (x**2 + y**2 <= aperture_radius**2).astype(float)

# Define steering angle
steering_angle_x = np.pi / 8  # Example: 45 degrees
steering_angle_y = np.pi / 8  # Example: 45 degrees

# Create phase mask (using 1j for imaginary unit)
phase_mask_x = np.exp(1j * k * np.sin(steering_angle_x) * x)
phase_mask_y = np.exp(1j * k * np.sin(steering_angle_y) * y)

phase_mask_x = rescale_matrix(np.real(phase_mask_x), 0, np.pi)
phase_mask_y = rescale_matrix(np.real(phase_mask_y), 0, np.pi)

grating_combined = (phase_mask_x + phase_mask_y) % np.pi

plt.figure()
plt.imshow(grating_combined, cmap='gray')
plt.colorbar()
plt.title('Real part of phase mask')
plt.show()



