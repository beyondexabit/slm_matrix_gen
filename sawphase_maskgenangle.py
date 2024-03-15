import numpy as np
import matplotlib.pyplot as plt 
from scipy.signal import sawtooth

# Function to rescale a matrix between a new min and max
def rescale_matrix(matrix, new_min, new_max):
    old_min, old_max = matrix.min(), matrix.max()
    return (new_max - new_min) / (old_max - old_min) * (matrix - old_min) + new_min


def sawtooth_myv(x, steering_angle_x):
    sawf = np.zeros_like(x)
    for n in range(1,30):
        #sawf += (-1**n) * np.real(np.exp(1j * k * np.sin(steering_angle_x) * n * x)/n)
        sawf += (-1**n) * (np.sin(n * x * np.sin(steering_angle_x)) / n)
    sawf = (0.5 - 1/np.pi*sawf)
    return sawf

# Define wavelength and wavenumber
wavelength = 1550e-9
k = 2 * np.pi / wavelength

# Create meshgrid coordinates
x = np.linspace(-20, 20, 1000)
y = np.linspace(-20, 20, 1000)
#x, y = np.meshgrid(x, y)  # Combine into meshgrid

# Define aperture radius and create aperture mask
#aperture_radius = 0.1
#aperture = (x**2 + y**2 <= aperture_radius**2).astype(float)

# Define steering angle
degrees = 45
steering_angle_x = np.radians(degrees)  # Example: 45 degrees
saw_phase_mask_x_my_v = sawtooth_myv(x, steering_angle_x)

degrees = 45
steering_angle_x = np.radians(degrees)  # Example: 45 degrees
saw_phase_mask_x = (sawtooth(np.sin(steering_angle_x) * x))


plt.figure()
plt.plot(x, saw_phase_mask_x_my_v)

plt.figure()
plt.plot(x, saw_phase_mask_x)


plt.show()
