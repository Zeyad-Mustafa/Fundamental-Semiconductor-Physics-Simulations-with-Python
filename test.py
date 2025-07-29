#!/usr/bin/env python3
"""
Comprehensive Test Suite for Semiconductor Physics Simulations
Tests for accuracy, edge cases, and performance
"""

import pytest
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock

# Add src directory to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import modules to test
from semiconductor_core import SemiconductorPhysics, PNJunction, BandDiagram, MATERIALS
from advanced_modules import QuantumEffects, TransportPhysics, DevicePhysics, NoiseAnalysis

class TestSemiconductorPhysics:
    """Test core semiconductor physics calculations"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.silicon = SemiconductorPhysics('Si')
        self.germanium = SemiconductorPhysics('Ge')
        self.gaas = SemiconductorPhysics('GaAs')
    
    def test_material_initialization(self):
        """Test material initialization"""
        assert self.silicon.material.name == 'Silicon'
        assert self.silicon.material.bandgap == 1.12
        assert self.silicon.material.epsilon_r == 11.68
        
        with pytest.raises(ValueError):
            SemiconductorPhysics('InvalidMaterial')
    
    def test_thermal_voltage(self):
        """Test thermal voltage calculation"""
        # At room temperature (300K), kT/q should be ~25.9 mV
        kT = self.silicon.thermal_voltage(300)
        assert abs(kT - 0.0259) < 0.001  # Within 1 mV
        
        # Test temperature scaling
        kT_200 = self.silicon.thermal_voltage(200)
        kT_400 = self.silicon.thermal_voltage(400)
        assert kT_400 > kT > kT_200
    
    def test_intrinsic_carrier_concentration(self):
        """Test intrinsic carrier concentration"""
        # Silicon at 300K should be ~1.45e10 cm^-3
        ni_300 = self.silicon.intrinsic_carrier_concentration(300)
        assert abs(ni_300 - 1.45e10) / 1.45e10 < 0.1  # Within 10%
        
        # Test temperature dependence (should increase with T)
        ni_250 = self.silicon.intrinsic_carrier_concentration(250)
        ni_350 = self.silicon.intrinsic_carrier_concentration(350)
        assert ni_350 > ni_300 > ni_250
        
        # Test material differences
        ni_ge = self.germanium.intrinsic_carrier_concentration(300)
        ni_gaas = self.gaas.intrinsic_carrier_concentration(300)
        assert ni_ge > ni_300 > ni_gaas  # Ge > Si > GaAs
    
    def test_carrier_concentrations_intrinsic(self):
        """Test carrier concentrations for intrinsic semiconductor"""
        n, p = self.silicon.carrier_concentrations(300, Na=0, Nd=0)
        ni = self.silicon.intrinsic_carrier_concentration(300)
        
        # For intrinsic: n = p = ni
        assert abs(n - ni) / ni < 0.01
        assert abs(p - ni) / ni < 0.01
        assert abs(n - p) / n < 0.01
    
    def test_carrier_concentrations_n_type(self):
        """Test carrier concentrations for N-type semiconductor"""
        Nd = 1e16
        n, p = self.silicon.carrier_concentrations(300, Na=0, Nd=Nd)
        ni = self.silicon.intrinsic_carrier_concentration(300)
        
        # For N-type: n ≈ Nd, p = ni²/Nd
        assert abs(n - Nd) / Nd < 0.01
        expected_p = ni**2 / Nd
        assert abs(p - expected_p) / expected_p < 0.01
        
        # Mass action law: n × p = ni²
        assert abs(n * p - ni**2) / ni**2 < 0.01
    
    def test_carrier_concentrations_p_type(self):
        """Test carrier concentrations for P-type semiconductor"""
        Na = 1e16
        n, p = self.silicon.carrier_concentrations(300, Na=Na, Nd=0)
        ni = self.silicon.intrinsic_carrier_concentration(300)
        
        # For P-type: p ≈ Na, n = ni²/Na
        assert abs(p - Na) / Na < 0.01
        expected_n = ni**2 / Na
        assert abs(n - expected_n) / expected_n < 0.01
        
        # Mass action law: n × p = ni²
        assert abs(n * p - ni**2) / ni**2 < 0.01
    
    def test_fermi_level(self):
        """Test Fermi level calculations"""
        # Intrinsic: Ef should be at Ei (0)
        Ef_intrinsic = self.silicon.fermi_level(300, Na=0, Nd=0)
        assert abs(Ef_intrinsic) < 1e-10
        
        # N-type: Ef should be above Ei
        Ef_n = self.silicon.fermi_level(300, Na=0, Nd=1e16)
        assert Ef_n > 0
        
        # P-type: Ef should be below Ei
        Ef_p = self.silicon.fermi_level(300, Na=1e16, Nd=0)
        assert Ef_p < 0
        
        # Higher doping should move Ef further from Ei
        Ef_n_high = self.silicon.fermi_level(300, Na=0, Nd=1e18)
        assert Ef_n_high > Ef_n
    
    def test_conductivity(self):
        """Test electrical conductivity calculations"""
        # Intrinsic conductivity should be low
        sigma_intrinsic = self.silicon.conductivity(300, Na=0, Nd=0)
        assert sigma_intrinsic < 1e-4  # Very low conductivity
        
        # Doped semiconductors should have higher conductivity
        sigma_n = self.silicon.conductivity(300, Na=0, Nd=1e16)
        sigma_p = self.silicon.conductivity(300, Na=1e16, Nd=0)
        
        assert sigma_n > sigma_intrinsic
        assert sigma_p > sigma_intrinsic
        
        # Higher doping should increase conductivity
        sigma_n_high = self.silicon.conductivity(300, Na=0, Nd=1e18)
        assert sigma_n_high > sigma_n
    
    def test_resistivity(self):
        """Test electrical resistivity calculations"""
        conductivity = self.silicon.conductivity(300, Na=0, Nd=1e16)
        resistivity = self.silicon.resistivity(300, Na=0, Nd=1e16)
        
        # Resistivity should be inverse of conductivity
        assert abs(resistivity * conductivity - 1) < 1e-10

class TestPNJunction:
    """Test P-N junction physics"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.pn_symmetric = PNJunction('Si', Na=1e16, Nd=1e16)
        self.pn_asymmetric = PNJunction('Si', Na=1e17, Nd=1e15)
    
    def test_built_in_voltage(self):
        """Test built-in voltage calculation"""
        Vbi = self.pn_symmetric.built_in_voltage(300)
        
        # For symmetric junction with moderate doping, Vbi should be ~0.7V
        assert 0.6 < Vbi < 0.8
        
        # Asymmetric junction should have different Vbi
        Vbi_asym = self.pn_asymmetric.built_in_voltage(300)
        assert abs(Vbi - Vbi_asym) > 0.01  # Should be different
        
        # Temperature dependence: Vbi should decrease with increasing T
        Vbi_250 = self.pn_symmetric.built_in_voltage(250)
        Vbi_350 = self.pn_symmetric.built_in_voltage(350)
        assert Vbi_250 > Vbi > Vbi_350
    
    def test_depletion_width(self):
        """Test depletion width calculation"""
        W, xn, xp = self.pn_symmetric.depletion_width(300)
        
        # Total width should be positive
        assert W > 0
        assert xn > 0
        assert xp > 0
        
        # Total width should equal sum of individual widths
        assert abs(W - (xn + xp)) < 1e-10
        
        # For symmetric junction: xn ≈ xp
        assert abs(xn - xp) / xn < 0.01
        
        # Reverse bias should increase depletion width
        W_reverse, _, _ = self.pn_symmetric.depletion_width(300, applied_voltage=-1)
        assert W_reverse > W
        
        # Forward bias should decrease depletion width
        W_forward, _, _ = self.pn_symmetric.depletion_width(300, applied_voltage=0.3)
        assert W_forward < W
    
    def test_current_density(self):
        """Test current density calculation"""
        # Zero voltage should give near-zero current
        J_zero = self.pn_symmetric.current_density(0, 300)
        assert abs(J_zero) < 1e-12
        
        # Forward bias should give positive current
        J_forward = self.pn_symmetric.current_density(0.5, 300)
        assert J_forward > 0
        
        # Reverse bias should give negative (small) current
        J_reverse = self.pn_symmetric.current_density(-0.5, 300)
        assert J_reverse < 0
        assert abs(J_reverse) < abs(J_forward)
        
        # Exponential behavior in forward bias
        J1 = self.pn_symmetric.current_density(0.6, 300)
        J2 = self.pn_symmetric.current_density(0.66, 300)  # 60 mV higher
        
        # Current should increase by factor of ~10 for 60 mV at 300K
        ratio = J2 / J1
        assert 8 < ratio < 12  # Approximately e^(0.06/0.026) ≈ 10
    
    def test_capacitance(self):
        """Test junction capacitance"""
        C = self.pn_symmetric.capacitance(300)
        assert C > 0
        
        # Reverse bias should decrease capacitance
        C_reverse = self.pn_symmetric.capacitance(300, applied_voltage=-1)
        assert C_reverse < C
        
        # Capacitance should scale with inverse square root of voltage
        C1 = self.pn_symmetric.capacitance(300, applied_voltage=-1)
        C2 = self.pn_symmetric.capacitance(300, applied_voltage=-4)
        
        # C ∝ 1/√V, so C1/C2 ≈ √(V2/V1) = 2
        ratio = C1 / C2
        assert 1.8 < ratio < 2.2

class TestQuantumEffects:
    """Test quantum mechanical calculations"""
    
    def setup_method(self):
        """Setup test fixtures"""
        from dataclasses import dataclass
        
        @dataclass
        class TestMaterial:
            bandgap = 1.42
            epsilon_r = 13.1
        
        self.material = TestMaterial()
        self.quantum = QuantumEffects(self.material)
    
    def test_fermi_dirac_distribution(self):
        """Test Fermi-Dirac distribution"""
        energy = np.array([0, 0.1, 0.2])  # eV
        fermi_energy = 0.1  # eV
        temperature = 300  # K
        
        fd = self.quantum.fermi_dirac_distribution(energy, fermi_energy, temperature)
        
        # At Fermi energy, f should be 0.5
        assert abs(fd[1] - 0.5) < 0.01
        
        # Below Fermi energy, f should be > 0.5
        assert fd[0] > 0.5
        
        # Above Fermi energy, f should be < 0.5
        assert fd[2] < 0.5
        
        # All values should be between 0 and 1
        assert np.all(fd >= 0)
        assert np.all(fd <= 1)
    
    def test_density_of_states_3d(self):
        """Test 3D density of states"""
        energy = np.linspace(0, 1, 100)  # eV
        effective_mass = 0.067  # GaAs electron mass
        band_edge = 0.0
        
        dos = self.quantum.density_of_states_3D(energy, effective_mass, band_edge)
        
        # DOS should be zero below band edge
        assert dos[0] == 0
        
        # DOS should increase as √E above band edge
        # Test scaling: DOS ∝ √E
        E1, E2 = 0.1, 0.4  # eV above band edge
        idx1 = np.argmin(np.abs(energy - E1))
        idx2 = np.argmin(np.abs(energy - E2))
        
        ratio = dos[idx2] / dos[idx1]
        expected_ratio = np.sqrt(E2 / E1)  # Should be 2
        assert abs(ratio - expected_ratio) / expected_ratio < 0.1
    
    def test_quantum_well_energy_levels(self):
        """Test quantum well energy level calculation"""
        well_width = 10  # nm
        barrier_height = 0.3  # eV
        effective_mass = 0.067
        
        levels = self.quantum.quantum_well_energy_levels(well_width, barrier_height, effective_mass)
        
        # Should have at least one level
        assert len(levels) > 0
        
        # All levels should be below barrier height
        assert all(E < barrier_height for E in levels)
        
        # Energy levels should increase
        for i in range(len(levels) - 1):
            assert levels[i+1] > levels[i]
        
        # First level should scale as 1/L²
        levels_5nm = self.quantum.quantum_well_energy_levels(5, barrier_height, effective_mass)
        levels_20nm = self.quantum.quantum_well_energy_levels(20, barrier_height, effective_mass)
        
        # E ∝ 1/L², so E(5nm)/E(20nm) = (20/5)² = 16
        if len(levels_5nm) > 0 and len(levels_20nm) > 0:
            ratio = levels_5nm[0] / levels_20nm[0]
            assert 14 < ratio < 18  # Approximately 16

class TestTransportPhysics:
    """Test transport phenomena"""
    
    def setup_method(self):
        """Setup test fixtures"""
        from dataclasses import dataclass
        
        @dataclass
        class TestMaterial:
            mu_n = 1350  # Silicon electron mobility
            mu_p = 480   # Silicon hole mobility
        
        self.material = TestMaterial()
        self.transport = TransportPhysics(self.material)
    
    def test_mobility_temperature_dependence(self):
        """Test mobility temperature dependence"""
        mu_300 = 1350  # cm²/V⋅s
        
        mu_200 = self.transport.mobility_temperature_dependence(200, mu_300)
        mu_400 = self.transport.mobility_temperature_dependence(400, mu_300)
        
        # Mobility should decrease with increasing temperature
        assert mu_200 > mu_300 > mu_400
        
        # Should follow T^-1.5 dependence
        expected_mu_200 = mu_300 * (300/200)**1.5
        assert abs(mu_200 - expected_mu_200) / expected_mu_200 < 0.01
    
    def test_diffusion_coefficient(self):
        """Test Einstein relation for diffusion coefficient"""
        mobility = 1350  # cm²/V⋅s
        temperature = 300  # K
        
        D = self.transport.diffusion_coefficient(mobility, temperature)
        
        # Einstein relation: D = μkT/q
        k_B = 1.38064852e-23  # J/K
        q = 1.602176634e-19   # C
        expected_D = mobility * k_B * temperature / q
        
        assert abs(D - expected_D) / expected_D < 0.01
    
    def test_drift_velocity(self):
        """Test drift velocity with saturation"""
        mobility = 1350  # cm²/V⋅s
        
        # Low field: v = μE
        E_low = 100  # V/cm
        v_low = self.transport.drift_velocity(E_low, mobility)
        expected_v_low = mobility * E_low
        assert abs(v_low - expected_v_low) / expected_v_low < 0.01
        
        # High field: velocity saturation
        E_high = 1e5  # V/cm
        v_high = self.transport.drift_velocity(E_high, mobility)
        v_sat = 1e7  # cm/s
        assert v_high < v_sat  # Should be saturated

class TestDevicePhysics:
    """Test advanced device physics"""
    
    def setup_method(self):
        """Setup test fixtures"""
        from dataclasses import dataclass
        
        @dataclass
        class TestMaterial:
            epsilon_r = 11.68
            bandgap = 1.12
        
        self.material = TestMaterial()
        self.device = DevicePhysics(self.material)
    
    def test_solar_cell_characteristics(self):
        """Test solar cell I-V characteristics"""
        results = self.device.solar_cell_characteristics(
            illumination=1.0, temperature=300, cell_area=100
        )
        
        # Check required fields
        required_fields = ['voltage', 'current', 'power', 'Voc', 'Jsc', 'Vmp', 'Jmp', 'efficiency']
        for field in required_fields:
            assert field in results
        
        # Physical constraints
        assert results['Voc'] > 0  # Open circuit voltage
        assert results['Jsc'] > 0  # Short circuit current
        assert results['Vmp'] > 0  # Maximum power voltage
        assert results['Jmp'] > 0  # Maximum power current
        assert 0 < results['efficiency'] < 100  # Efficiency percentage
        
        # Maximum power point should be less than Voc and Jsc
        assert results['Vmp'] < results['Voc']
        assert results['Jmp'] < results['Jsc']
    
    def test_bjt_characteristics(self):
        """Test BJT characteristics"""
        base_currents = np.array([1e-6, 2e-6, 3e-6])  # A
        results = self.device.bjt_characteristics(
            base_currents, temperature=300, beta=100
        )
        
        # Check required fields
        assert 'collector_current' in results
        assert 'collector_voltage' in results
        assert 'beta' in results
        
        # Current gain check
        Ic = results['collector_current']
        Ib = results['base_current']
        
        # In active region, Ic ≈ β × Ib
        for i, ib in enumerate(Ib):
            ic_active = Ic[i, -1]  # High Vce value (active region)
            beta_calculated = ic_active / ib
            assert abs(beta_calculated - 100) / 100 < 0.2  # Within 20%

class TestPerformance:
    """Performance and benchmark tests"""
    
    def test_calculation_speed(self):
        """Test computational performance"""
        import time
        
        silicon = SemiconductorPhysics('Si')
        
        # Time intrinsic carrier concentration calculation
        start_time = time.time()
        for _ in range(1000):
            silicon.intrinsic_carrier_concentration(300)
        calc_time = time.time() - start_time
        
        # Should complete 1000 calculations in under 1 second
        assert calc_time < 1.0
        print(f"1000 ni calculations took {calc_time:.3f} seconds")
    
    def test_memory_usage(self):
        """Test memory efficiency"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large arrays for calculation
        temperatures = np.linspace(200, 500, 10000)
        silicon = SemiconductorPhysics('Si')
        
        ni_values = [silicon.intrinsic_carrier_concentration(T) for T in temperatures]
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 100 MB for this test)
        assert memory_increase < 100
        print(f"Memory increase: {memory_increase:.1f} MB")

class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_extreme_temperatures(self):
        """Test behavior at extreme temperatures"""
        silicon = SemiconductorPhysics('Si')
        
        # Very low temperature
        ni_low = silicon.intrinsic_carrier_concentration(50)  # 50K
        assert ni_low > 0
        assert ni_low < 1e-10  # Should be very small
        
        # Very high temperature
        ni_high = silicon.intrinsic_carrier_concentration(1000)  # 1000K
        assert ni_high > 0
        assert ni_high > 1e15  # Should be large
    
    def test_extreme_doping(self):
        """Test behavior with extreme doping concentrations"""
        silicon = SemiconductorPhysics('Si')
        
        # Very low doping
        n, p = silicon.carrier_concentrations(300, Na=0, Nd=1e10)
        ni = silicon.intrinsic_carrier_concentration(300)
        assert n > ni  # Should still be n-type
        
        # Very high doping
        n, p = silicon.carrier_concentrations(300, Na=0, Nd=1e20)
        assert n > 1e19  # Should approach doping concentration
    
    def test_
