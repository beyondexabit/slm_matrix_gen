import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift, ifft2 # Python DFT


def radial_zernike(rho, n, m):
    """
    Compute the radial component of the Zernike polynomial.

    Parameters:
        rho (numpy.ndarray): Radial coordinate scaled to the range [0, 1].
        n (int): Radial degree of the Zernike polynomial.
        m (int): Azimuthal frequency of the Zernike polynomial.

    Returns:
        numpy.ndarray: Values of the radial component of the Zernike polynomial.
    """
    if (n - abs(m)) % 2 != 0 or abs(m) > n:
        return np.zeros_like(rho)

    pre_factor = np.sqrt((2 * (n + 1)) / (np.pi))
    poly_coefficients = np.zeros(n + 1)
    poly_coefficients[n - abs(m)] = 1

    radial_poly = np.polynomial.legendre.Legendre(poly_coefficients)
    return pre_factor * radial_poly(np.sqrt(2) * rho)


def angular_zernike(theta, m):
    """
    Compute the angular component of the Zernike polynomial.

    Parameters:
        theta (numpy.ndarray): Angular coordinate in radians.
        m (int): Azimuthal frequency of the Zernike polynomial.

    Returns:
        numpy.ndarray: Values of the angular component of the Zernike polynomial.
    """
    return np.exp(1j * m * theta)


def generate_zernike_mask(size, radius, polynomial_order):
    """
    Generate a Zernike polynomial mask.

    Parameters:
        size (int): Size of the square mask.
        radius (float): Radius of the circle inside the square mask.
        polynomial_order (int): The order of the Zernike polynomial.

    Returns:
        numpy.ndarray: A 2D array representing the Zernike polynomial mask.
    """
    x = np.linspace(-radius, radius, size)
    y = np.linspace(-radius, radius, size)
    xx, yy = np.meshgrid(x, y)
    rr = np.sqrt(xx**2 + yy**2)

    mask = np.zeros((size, size))

    # Mask inside the circle
    mask[rr <= radius] = 1

    # Normalize coordinates to range [0, 1]
    rho = rr / radius
    theta = np.arctan2(yy, xx) % (2 * np.pi)  # Ensure angle is in [0, 2pi)

    # Generate Zernike polynomial
    zernike_values = np.zeros_like(rr, dtype=np.complex128)
    for n in range(polynomial_order + 1):
        for m in range(-n, n + 1, 2):
            if abs(m) <= n:
                radial = radial_zernike(rho, n, m)
                angular = angular_zernike(theta, m)
                zernike_values += radial * angular

    # Apply Zernike polynomial to the mask
    mask *= np.abs(zernike_values)

    return mask

# Example usage
size = 200  # Size of the square mask
radius = 0.5  # Radius of the circle inside the square mask

#polynomial_order = 2  # Order of the Zernike polynomial

for i in range(10):
    zernike_mask = generate_zernike_mask(size, radius, i)
    plt.figure()
    plt.imshow(zernike_mask, cmap='gray')
    plt.title(f"Zernike Polynomial Mask (Order {i})")
    plt.colorbar()
    F = fft2(zernike_mask)                        
    F = fftshift(F)
    P = np.abs(F)
    #log_scaled_image = np.log(1 + P)

    plt.figure()
    plt.imshow(P, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title('Magnitude of Fourier Transform')


plt.show()


