import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import sawtooth
import sys
from numpy.fft import fft2, fftshift, ifft2 # Python DFT


grating_frequency_x = 20
macropixel_size = 100

x = np.linspace(0, 2 * np.pi * grating_frequency_x, macropixel_size)
#grating_x = (sawtooth(x))

#x = np.linspace(0, 216, 217)
grating_x = (sawtooth(x))

wavelength = 1550e-9
grating_period = 20 * 8e-6
degrees = np.degrees(np.arcsin(wavelength / grating_period))
print(degrees)

focal_length = 0.1
xdistance = np.tan(np.deg2rad(degrees))*focal_length
print(xdistance)


x = np.linspace(0, 216, 217)
degrees = 45
steering_angle_x = np.radians(degrees)  # Example: 45 degrees
saw_phase_mask_x = (sawtooth(np.sin(steering_angle_x) * x))



plt.figure()
plt.plot(grating_x)

plt.figure()
plt.plot(saw_phase_mask_x)

#plt.show()


