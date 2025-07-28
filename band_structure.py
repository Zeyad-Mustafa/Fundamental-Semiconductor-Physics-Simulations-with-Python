import numpy as np
import matplotlib.pyplot as plt

# --- Constants ---
m_e = 9.109e-31  # Electron rest mass (kg)
hbar = 1.054e-34  # Reduced Planck constant (J.s)
eV_to_J = 1.602e-19 # Conversion factor from eV to Joules

# --- Free Electron Model ---
def free_electron_energy(k):
    """Calculates energy for the free electron model."""
    return (hbar**2 * k**2) / (2 * m_e)

def plot_free_electron_band():
    """Plots the E-k diagram for the free electron model."""
    k_values = np.linspace(-5e9, 5e9, 500) # Wavevector range (m^-1)
    energy_joules = free_electron_energy(k_values)
    energy_eV = energy_joules / eV_to_J

    plt.figure(figsize=(8, 6))
    plt.plot(k_values, energy_eV, label=r'$E = \frac{\hbar^2 k^2}{2m_e}$')
    plt.xlabel('Wavevector k ($m^{-1}$)')
    plt.ylabel('Energy E (eV)')
    plt.title('Free Electron Energy Band')
    plt.grid(True)
    plt.legend()
    plt.show()

# --- Kronig-Penney Model ---
def kronig_penney_lhs(E_eV, V0_eV, a_nm, b_nm):
    """
    Calculates the left-hand side of the Kronig-Penney equation.
    E_eV: Energy in eV
    V0_eV: Potential barrier height in eV
    a_nm: Well width in nm
    b_nm: Barrier width in nm
    """
    E = E_eV * eV_to_J  # Convert energy to Joules
    V0 = V0_eV * eV_to_J # Convert potential to Joules
    a = a_nm * 1e-9     # Convert nm to meters
    b = b_nm * 1e-9     # Convert nm to meters

    # Handle cases where E > V0 (alpha is real, beta is imaginary)
    # and E < V0 (alpha is real, beta is real)
    if E < 0: # Energy must be non-negative for this model's common application
        return np.nan # Or handle as appropriate for your specific model extension

    alpha = np.sqrt(2 * m_e * E) / hbar

    if E < V0:
        beta = np.sqrt(2 * m_e * (V0 - E)) / hbar
        P = (m_e * V0 * b) / (hbar**2) # Strength of the barrier
        lhs = P * (np.sin(alpha * a) / (alpha * a)) + np.cos(alpha * a) * np.cosh(beta * b) # Corrected for E < V0
        # Note: The common textbook form P * sin(alpha*a)/(alpha*a) + cos(alpha*a) is for delta-function potential.
        # For finite rectangular barrier, it's more complex. A common approximation for E < V0 is:
        # cos(ka) = cosh(beta*b)cos(alpha*a) - ((alpha^2 - beta^2)/(2*alpha*beta))sinh(beta*b)sin(alpha*a)
        # Let's use a simplified form often seen for the general concept of bands:
        # A common simplified form for the Kronig-Penney model (often for delta function limit) is:
        # P * sin(alpha*a)/(alpha*a) + cos(alpha*a) = cos(Ka)
        # For a finite barrier, the equation is more involved.
        # Let's use a more general form often used to illustrate band gaps:
        # cos(ka) = cos(alpha*a)cos(beta*b) - ( (alpha^2+beta^2)/(2*alpha*beta) ) * sin(alpha*a)sin(beta*b)
        # However, for simply showing allowed/forbidden bands by plotting LHS vs E,
        # the simpler form P * sin(alpha*a)/(alpha*a) + cos(alpha*a) is often used to illustrate the concept.
        # Let's use the simplified form to illustrate the concept of allowed bands for E < V0
        # where P is the barrier strength.
        # A common form for finite barriers:
        # f(E) = cos(alpha*a) * cosh(beta*b) + ( (beta^2 - alpha^2) / (2*alpha*beta) ) * sin(alpha*a) * sinh(beta*b)
        # We want to plot f(E) and check if it's between -1 and 1.

        # Let's use a common simplified form to illustrate the concept of allowed bands for E < V0
        # where P is the barrier strength. This form is for the delta-function limit, but
        # it's often used conceptually.
        # P = (m_e * V0 * b) / (hbar**2)
        # lhs = P * (np.sin(alpha * a) / (alpha * a)) + np.cos(alpha * a)
        # For a finite barrier where E < V0, a more accurate simplified form for plotting allowed bands is:
        # cos(Ka) = cos(alpha*a)cosh(beta*b) + ((alpha^2 - beta^2)/(2*alpha*beta))sin(alpha*a)sinh(beta*b)
        # Let's use this for better physical representation.
        lhs = np.cos(alpha * a) * np.cosh(beta * b) + \
              ((alpha**2 - beta**2) / (2 * alpha * beta)) * np.sin(alpha * a) * np.sinh(beta * b)
    else: # E > V0 (electrons can pass over the barrier)
        gamma = np.sqrt(2 * m_e * (E - V0)) / hbar
        # For E > V0, the equation becomes:
        # cos(Ka) = cos(alpha*a)cos(gamma*b) - ((alpha^2 + gamma^2)/(2*alpha*gamma))sin(alpha*a)sin(gamma*b)
        lhs = np.cos(alpha * a) * np.cos(gamma * b) - \
              ((alpha**2 + gamma**2) / (2 * alpha * gamma)) * np.sin(alpha * a) * np.sin(gamma * b)

    return lhs

def plot_kronig_penney_bands(V0_eV=10, a_nm=0.5, b_nm=0.1):
    """
    Plots the Kronig-Penney energy bands by evaluating the LHS of the equation.
    Allowed bands are where LHS is between -1 and 1.
    """
    energies_eV = np.linspace(0.1, 20, 500) # Energy range in eV
    lhs_values = np.array([kronig_penney_lhs(E, V0_eV, a_nm, b_nm) for E in energies_eV])

    plt.figure(figsize=(10, 7))
    plt.plot(energies_eV, lhs_values, label='LHS of Kronig-Penney Equation')
    plt.axhline(y=1, color='r', linestyle='--', label='Allowed Region Boundary (+1)')
    plt.axhline(y=-1, color='r', linestyle='--', label='Allowed Region Boundary (-1)')
    plt.fill_between(energies_eV, -1, 1, where=(lhs_values >= -1) & (lhs_values <= 1),
                     color='green', alpha=0.2, label='Allowed Energy Bands')

    plt.xlabel('Energy E (eV)')
    plt.ylabel('f(E) = cos(Ka)')
    plt.title(f'Kronig-Penney Model: $V_0={V0_eV}$ eV, $a={a_nm}$ nm, $b={b_nm}$ nm')
    plt.ylim(-5, 5) # Adjust y-limit for better visualization of bands
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    print("Plotting Free Electron Energy Band...")
    plot_free_electron_band()

    print("\nPlotting Kronig-Penney Energy Bands...")
    plot_kronig_penney_bands()
    # You can experiment with different parameters:
    # plot_kronig_penney_bands(V0_eV=5, a_nm=0.8, b_nm=0.2)
    # plot_kronig_penney_bands(V0_eV=15, a_nm=0.3, b_nm=0.05)
