import numpy as np
import matplotlib.pyplot as plt 



def sawtooth(x, steering_angle_x):
    sawf = np.zeros_like(x)
    for n in range(1,30):
        #sawf += (-1**n) * np.real(np.exp(1j * k * np.sin(steering_angle_x) * n * x)/n)
        sawf += (-1**n) * (np.sin(2*np.pi * n * x * np.sin(steering_angle_x)) / n)
    sawf = (0.5 - 1/np.pi*sawf)
    return sawf

wavelength = 1550e-9
k = 2 * np.pi / wavelength
degrees = 45
steering_angle_x = np.radians(degrees)  # Example: 45 degrees

x = np.arange(-5, 5.01, 0.01)
sin = np.exp(1j * k * np.sin(steering_angle_x) * x)
saw = np.sin( x * np.sin(steering_angle_x) * k) 


plt.figure()
plt.plot(x, sin)

plt.figure()
plt.plot(x, saw)

# Fast Fourier Transform (FFT) for frequency analysis
fft_sin = np.fft.fft(sin)
fft_saw = np.fft.fft(saw)

plt.figure()
plt.plot(x, fft_sin)

plt.figure()
plt.plot(x, fft_saw)


plt.show()

