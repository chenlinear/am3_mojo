"""Port of RunParams: parameters describing the radiation zone."""

from am3_mojo.utils.constants import *
from am3_mojo.utils.libm import *


struct RunParams(Copyable, Movable):
    var mag_field: Float64
    var escape_timescale: Float64
    var escape_fraction_pairs: Float64
    var escape_fraction_photons: Float64
    var escape_fraction_protons: Float64
    var escape_fraction_neutrons: Float64
    var escape_fraction_pions: Float64
    var escape_fraction_muons: Float64
    var escape_fraction_neutrinos: Float64
    var expansion_timescale: Float64
    var electron_accel_factor_for_emax_estimate: Float64
    var proton_accel_factor_for_emax_estimate: Float64
    var estimated_electron_emax: Float64
    var estimated_proton_emax: Float64
    var pp_target_proton_density: Float64
    var powerlaw_injection_electrons_volume: Float64
    var powerlaw_injection_electrons_luminosity: Float64
    var powerlaw_injection_electrons_emin: Float64
    var powerlaw_injection_electrons_ebreak: Float64
    var powerlaw_injection_electrons_emax: Float64
    var powerlaw_injection_electrons_index: Float64
    var powerlaw_injection_electrons_index_above_break: Float64
    var powerlaw_injection_electrons_cutoff_steepness: Float64
    var powerlaw_injection_protons_volume: Float64
    var powerlaw_injection_protons_luminosity: Float64
    var powerlaw_injection_protons_emin: Float64
    var powerlaw_injection_protons_ebreak: Float64
    var powerlaw_injection_protons_emax: Float64
    var powerlaw_injection_protons_index: Float64
    var powerlaw_injection_protons_index_above_break: Float64
    var powerlaw_injection_protons_cutoff_steepness: Float64
    var dt: Float64
    var REG: Float64

    def __init__(out self):
        self.mag_field = 0.01
        self.escape_timescale = 1e6
        self.escape_fraction_pairs = 1.0
        self.escape_fraction_photons = 1.0
        self.escape_fraction_protons = 1.0
        self.escape_fraction_neutrons = 1.0
        self.escape_fraction_pions = 1.0
        self.escape_fraction_muons = 1.0
        self.escape_fraction_neutrinos = 1.0
        self.expansion_timescale = self.escape_timescale
        self.electron_accel_factor_for_emax_estimate = 1.0
        self.proton_accel_factor_for_emax_estimate = 1.0
        self.estimated_electron_emax = 1e10
        self.estimated_proton_emax = 1e15
        self.pp_target_proton_density = 0.0
        var vol = 4.0 / 3.0 * PI * c_pow(self.escape_timescale * C_LIGHT, 3.0)
        self.powerlaw_injection_electrons_volume = vol
        self.powerlaw_injection_electrons_luminosity = 0.0
        self.powerlaw_injection_electrons_emin = 1e7
        self.powerlaw_injection_electrons_ebreak = 1e7
        self.powerlaw_injection_electrons_emax = 1e13
        self.powerlaw_injection_electrons_index = 2.0
        self.powerlaw_injection_electrons_index_above_break = 2.0
        self.powerlaw_injection_electrons_cutoff_steepness = 1.0
        self.powerlaw_injection_protons_volume = vol
        self.powerlaw_injection_protons_luminosity = 0.0
        self.powerlaw_injection_protons_emin = 1e10
        self.powerlaw_injection_protons_ebreak = 1e10
        self.powerlaw_injection_protons_emax = 1e15
        self.powerlaw_injection_protons_index = 2.0
        self.powerlaw_injection_protons_index_above_break = 2.0
        self.powerlaw_injection_protons_cutoff_steepness = 1.0
        self.dt = 1e-3 * self.escape_timescale
        self.REG = 1.0e-50

    def print_params(self):
        print("AM3 params (comoving):")
        print(
            "escape_timescale: ",
            self.escape_timescale,
            " s\n  with fractions: (",
            "protons: ",
            self.escape_fraction_protons,
            ", ",
            "neutrons: ",
            self.escape_fraction_neutrons,
            ", ",
            "pions: ",
            self.escape_fraction_pions,
            ", ",
            "muons: ",
            self.escape_fraction_muons,
            ", ",
            "neutrinos: ",
            self.escape_fraction_neutrinos,
            ", ",
            "pairs: ",
            self.escape_fraction_pairs,
            ")",
            sep="",
        )
        print("expansion_timescale: ", self.expansion_timescale, " s", sep="")
        print("t_adi: ", 3.0 * self.expansion_timescale, " s", sep="")
        print("B: ", self.mag_field, " G", sep="")
        print(
            "pp target density = ",
            self.pp_target_proton_density,
            " cm^-3",
            sep="",
        )
        print("dt = ", self.dt, " s", sep="")
        print(
            "eta_Bohm: electrons:",
            self.electron_accel_factor_for_emax_estimate,
            ", protons: ",
            self.proton_accel_factor_for_emax_estimate,
            sep="",
        )
        print(
            "inj. el:\n  power density = ",
            self.powerlaw_injection_electrons_luminosity
            / self.powerlaw_injection_electrons_volume,
            " erg/s/cm^3, \n   Emin = ",
            self.powerlaw_injection_electrons_emin,
            " eV,\n   Ebreak = ",
            self.powerlaw_injection_electrons_ebreak,
            " eV,\n   Emax = ",
            self.powerlaw_injection_electrons_emax,
            " eV,\n   index = ",
            self.powerlaw_injection_electrons_index,
            ",\n   index above break = ",
            self.powerlaw_injection_electrons_index_above_break,
            ",\n   cut_off steepness = ",
            self.powerlaw_injection_electrons_cutoff_steepness,
            sep="",
        )
        print(
            "inj. protons: power density = ",
            self.powerlaw_injection_protons_luminosity
            / self.powerlaw_injection_protons_volume,
            " erg/s/cm^3, \n   Emin = ",
            self.powerlaw_injection_protons_emin,
            " eV,\n   Ebreak = ",
            self.powerlaw_injection_protons_ebreak,
            " eV,\n   Emax = ",
            self.powerlaw_injection_protons_emax,
            " eV,\n   index = ",
            self.powerlaw_injection_protons_index,
            ",\n   index above break = ",
            self.powerlaw_injection_protons_index_above_break,
            ",\n   cut_off steepness = ",
            self.powerlaw_injection_protons_cutoff_steepness,
            "\n",
            sep="",
        )
