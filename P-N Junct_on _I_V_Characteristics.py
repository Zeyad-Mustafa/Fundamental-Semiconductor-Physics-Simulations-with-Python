from semiconductor_physics import PNJunction
import numpy as np

# Create P-N junction
pn = PNJunction('Si', Na=1e16, Nd=1e16)

# Voltage range
voltages = np.linspace(-1, 1, 1000)

# Calculate current density
currents = [pn.current_density(V, 300) for V in voltages]

# Plot I-V curve
plt.figure(figsize=(10, 6))
plt.semilogy(voltages, np.abs(currents), 'r-', linewidth=2)
plt.xlabel('Voltage (V)')
plt.ylabel('Current Density (A/cm²)')
plt.title('P-N Junction I-V Characteristic')
plt.grid(True, alpha=0.3)
plt.show()
