import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import sawtooth
import sys
from numpy.fft import fft2, fftshift, ifft2 # Python DFT


grating_frequency_x = 20
macropixel_size = 216

x = np.linspace(0, 2 * np.pi * grating_frequency_x, macropixel_size) #oscillations per macropixel
# PERIOD = macropixel_size / grating_frequency_x

#x = np.linspace(0, 216, 217)
grating_x = (sawtooth(x))
wavelength = 1550e-9

grating_period = macropixel_size/grating_frequency_x * 8e-6
#print(grating_period)
degrees = np.degrees(np.arcsin(wavelength / grating_period))
print('degrees:',degrees)

focal_length = 0.1
xdistance = np.tan(np.deg2rad(degrees))*focal_length
print('displacement:', xdistance)

'''
x = np.linspace(0, 216, 217)
degrees = 45
steering_angle_x = np.radians(degrees)  # Example: 45 degrees
saw_phase_mask_x = (sawtooth(np.sin(steering_angle_x) * x))
'''

# real angle of propagation assuming misplaced hologram
displacement_from_axis = 1.296e-3 # for x axis, left is positive and right is negative
                          # for y axis the displacement doesnt matter as the frequencies 
true_angle_prop = degrees + np.arctan((displacement_from_axis)/focal_length)
print('true_angle_prop:',np.degrees(true_angle_prop))

plt.figure()
plt.plot(grating_x)

#plt.figure()
#plt.plot(saw_phase_mask_x)
#plt.show()



#                                   y,  x
#Center element of grating_combined:324,798
#Center element of grating_combined:324,1122
#Center element of grating_combined:540,798
#Center element of grating_combined:540,112



# For x spatial displacement from centre of SLM:
# ==========================================================================================================
# 798pixels = 1920/2 - 798 = 162pixels = 162*8e-6 = -1.296e-3m, Angle of propagation (gfx = 20) 
#degrees: 1.0279308168512793
#displacement: 0.0017942702359514378
#true_angle_prop: 1.7700311178216186
# At focal length 0.1m i calculate = 0.00178m and measure = 0.00178, holo from centre

# at image plane centre coordinate = 0.00178m
# For 20 cm of further propagation coordinate is =0.01172
# From calculations I get: degrees = 2.862



# 1122pixels = 1920/2 - 1122 = -162pixels = -162*8e-6 = 1.296e-3m, Angle of propagation (gfx = 40)
#degrees: 2.0561926818224703
#displacement: 0.003590274673337588
#true_angle_prop: 1.314291989491067
# At focal length 0.1m i calculate = 0.00359m and measure = 0.00359, holo from centre



# For y spatial displacement from centre of SLM:
# ==========================================================================================================
# 324pixels = 1080/2 - 324 = 216pixels = 216*8e-6 = -1.728e-3m, Angle of propagation (gfy = -46)= -1.3748128461363822
#degrees: -2.3647853892280315
#displacement: -0.004129674326343853
#true_angle_prop: -3.3523694130649715

# 540pixels = 1080/2 - 540 = 0pixels = 0*8e-6 = 0m, Angle of propagation (gfy = 0) = 0.0

# 751pixels = 1080/2 - 751 = -216pixels = -216*8e-6 = 1.728e-3m, Angle of propagation (gfy = 46) = 1.3748128461363822
#degrees: 2.3647853892280315
#displacement: 0.004129674326343853
#true_angle_prop: 3.3523694130649715




