import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.fft import fft2, ifft2, fftfreq
import matplotlib

# Set random seed for reproducibility
np.random.seed(0)

# Simulation Parameters
Lx, Ly = 1.0, 1.0       # Domain size
Nx, Ny = 64, 64         # Number of grid points
dx, dy = Lx / Nx, Ly / Ny
dt = 0.001              # Time step
nt = 500                # Number of time steps

# Physical Parameters
nu = 1e-5               # Viscosity
kappa = 1e-5            # Thermal diffusivity
g = 1.0                 # Gravity
theta0 = 1.0            # Reference temperature

# Create grid
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Initialize Fields
omega = np.zeros((Ny, Nx))          # Vorticity
theta = np.zeros((Ny, Nx))          # Temperature
psi = np.zeros((Ny, Nx))            # Streamfunction

# Initial Temperature Perturbation: Gaussian hill in the center
sigma = 0.1
theta += 0.5 * np.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / (2 * sigma**2))

# Function to solve Poisson equation using FFT
def solve_poisson_fft(rhs):
    # Fourier transform of the right-hand side
    rhs_hat = fft2(rhs)
    
    # Wave numbers
    kx = fftfreq(Nx, d=dx) * 2 * np.pi
    ky = fftfreq(Ny, d=dy) * 2 * np.pi
    kx[0] = 1e-6  # Prevent division by zero
    ky[0] = 1e-6
    
    KX, KY = np.meshgrid(kx, ky)
    K_squared = KX**2 + KY**2
    K_squared[K_squared == 0] = 1e-6  # Prevent division by zero
    
    # Solve in Fourier space
    psi_hat = rhs_hat / K_squared
    
    # Inverse Fourier transform to get streamfunction
    psi = np.real(ifft2(psi_hat))
    return psi

# Function to compute velocity from streamfunction
def compute_velocity(psi):
    u =  np.gradient(psi, dy, axis=0)  # dψ/dy
    v = -np.gradient(psi, dx, axis=1)  # -dψ/dx
    return u, v

# Function to compute advection terms using central differences
def advection(f, u, v):
    f_x = (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2 * dx)
    f_y = (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2 * dy)
    return u * f_x + v * f_y

# Animation Setup
fig, ax = plt.subplots(figsize=(6,5))
contour = ax.contourf(X, Y, omega, levels=50, cmap='RdBu_r')
plt.colorbar(contour)
ax.set_title('Vorticity and Temperature')
ax.set_xlabel('x')
ax.set_ylabel('y')

def animate(frame):
    global omega, theta, psi
    for _ in range(5):  # Update multiple times per frame for stability
        # Compute streamfunction from vorticity
        psi = solve_poisson_fft(-omega)
        
        # Compute velocities
        u, v = compute_velocity(psi)
        
        # Update vorticity
        adv_omega = advection(omega, u, v)
        diffusion_omega = nu * (np.roll(omega, -1, axis=0) + np.roll(omega, 1, axis=0) +
                                np.roll(omega, -1, axis=1) + np.roll(omega, 1, axis=1) - 4 * omega) / dx**2
        buoyancy = (g / theta0) * (np.roll(theta, -1, axis=1) - np.roll(theta, 1, axis=1)) / (2 * dx)
        omega_new = omega + dt * (-adv_omega + diffusion_omega + buoyancy)
        
        # Update temperature
        adv_theta = advection(theta, u, v)
        diffusion_theta = kappa * (np.roll(theta, -1, axis=0) + np.roll(theta, 1, axis=0) +
                                    np.roll(theta, -1, axis=1) + np.roll(theta, 1, axis=1) - 4 * theta) / dx**2
        theta_new = theta + dt * (-adv_theta + diffusion_theta)
        
        # Update fields
        omega, theta = omega_new, theta_new
        
        # Apply periodic boundary conditions
        omega = np.roll(omega, 1, axis=0)
        omega = np.roll(omega, 1, axis=1)
        theta = np.roll(theta, 1, axis=0)
        theta = np.roll(theta, 1, axis=1)
    
    # Update plot
    ax.clear()
    cont = ax.contourf(X, Y, omega, levels=50, cmap='RdBu_r')
    ax.set_title(f'Vorticity at frame {frame}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    return cont.collections

# Choose writer based on availability
def get_writer():
    # Try to use ffmpeg
    try:
        Writer = animation.writers['ffmpeg']
        return Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
    except (KeyError, RuntimeError):
        # Fallback to imagemagick
        try:
            Writer = animation.writers['imagemagick']
            return Writer(fps=30)
        except (KeyError, RuntimeError):
            print("Neither ffmpeg nor imagemagick is available. Animation will not be saved.")
            return None

writer = get_writer()

if writer:
    ani = animation.FuncAnimation(fig, animate, frames=nt//5, interval=50, blit=False)
    # Save the animation
    ani.save('vorticity_convection.mp4', writer=writer)
else:
    # Just show the animation
    ani = animation.FuncAnimation(fig, animate, frames=nt//5, interval=50, blit=False)
    plt.show()
