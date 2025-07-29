import numpy as np
import matplotlib.pyplot as plt
from semiconductor_physics import SemiconductorPhysics

# Create silicon semiconductor
si = SemiconductorPhysics('Si')

# Temperature range
temperatures = np.linspace(250, 400, 100)

# Calculate intrinsic carrier concentration
ni_values = [si.intrinsic_carrier_concentration(T) for T in temperatures]

# Plot results
plt.figure(figsize=(10, 6))
plt.semilogy(temperatures, ni_values, 'b-', linewidth=2)
plt.xlabel('Temperature (K)')
plt.ylabel('Intrinsic Carrier Concentration (cm⁻³)')
plt.title('Temperature Dependence of n_i in Silicon')
plt.grid(True, alpha=0.3)
plt.show()
