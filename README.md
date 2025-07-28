# Fundamental-Semiconductor-Physics-Simulations-with-Python
To visualize the energy band structure of electrons in a periodic potential, starting with the simple free electron model and then moving to the more realistic Kronig-Penney model. This will help understand the formation of energy bands and band gaps.
Physics Concepts
Free Electron Model:

Assumes electrons move freely in a constant potential.

Energy (E) is purely kinetic: E=
frachbar 
2
 k 
2
 2m_e, where 
hbar is the reduced Planck constant, k is the wavevector, and m_e is the electron rest mass.

The E-k diagram is a parabola.

Kronig-Penney Model:

Models a 1D crystal as a series of rectangular potential wells (or barriers).

Introduces the concept of allowed energy bands and forbidden energy gaps due to the interaction of electrons with the periodic potential.

The solution involves solving the Schrödinger equation for the periodic potential, leading to a transcendental equation that relates energy to the wavevector k:


P 
αa
sin(αa)
​
 +cos(αa)=cos(ka)

where:

alpha=
sqrtfrac2m_eEhbar 
2
 

beta=
sqrtfrac2m_e(V_0−E)hbar 
2
  (for E\<V_0)

P=
fracm_eV_0bhbar 
2
  (strength of the barrier)

a is the width of the well, b is the width of the barrier, so (a+b) is the period.

V_0 is the height of the potential barrier.

The allowed energy values correspond to regions where the left-hand side of the equation is between -1 and 1.

Python Approach
Libraries: NumPy for numerical calculations (arrays, square roots, exponentials, sinc function if needed), Matplotlib for plotting.

Free Electron: Straightforward calculation and plotting of a parabola.

Kronig-Penney:

Define the parameters (a,b,V_0,m_e,
hbar).

Iterate through a range of energy values.

For each energy, calculate the left-hand side (LHS) of the Kronig-Penney equation.

Plot LHS vs. Energy. The allowed bands are where LHS is between -1 and 1.

Alternatively, you can plot k vs. E by solving the equation for k (which is more complex as it involves inverse cosine). A simpler approach for visualization is to plot the LHS and visually identify the allowed regions.
