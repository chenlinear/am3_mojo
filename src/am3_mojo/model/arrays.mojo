"""Port of AM3Arrays."""

from std.math import floor
from am3_mojo.model.particles import *
from am3_mojo.utils.buffers import *
from am3_mojo.utils.constants import *
from am3_mojo.utils.libm import *


struct AM3Arrays(Copyable, Movable):
    var dlnE: Float64
    var nbins_lep: Int
    var egrid_lep: Vec
    var nbins_pho: Int
    var ind_min_energy_pho: Int
    var egrid_pho: Vec
    var nbins_had: Int
    var egrid_had: Vec
    var nbins_nu: Int
    var ind_min_energy_nu: Int
    var egrid_nu: Vec
    var nbins_lep_hadgrid: Int
    var ind_min_energy_lep_hadgrid: Int
    var egrid_had_leptons: Vec
    var nbins_pho_hadgrid: Int
    var ind_min_energy_pho_hadgrid: Int
    var egrid_had_photons: Vec

    var n: List[Vec]
    var h2l_lo: VecI
    var h2l_w: Vec
    var l2h_lo: VecI
    var l2h_w: Vec

    def __init__(out self):
        self.dlnE = 0.1
        self.nbins_pho = 620
        self.ind_min_energy_pho = 280
        self.nbins_lep = self.nbins_pho - self.ind_min_energy_pho
        self.nbins_had = self.nbins_lep - 26
        self.ind_min_energy_nu = 150
        self.nbins_nu = self.nbins_had + self.ind_min_energy_nu
        self.nbins_lep_hadgrid = self.nbins_lep
        self.nbins_pho_hadgrid = self.nbins_pho
        self.ind_min_energy_lep_hadgrid = 76
        self.ind_min_energy_pho_hadgrid = (
            self.ind_min_energy_lep_hadgrid + self.ind_min_energy_pho
        )
        self.egrid_lep = Vec()
        self.egrid_pho = Vec()
        self.egrid_had = Vec()
        self.egrid_nu = Vec()
        self.egrid_had_leptons = Vec()
        self.egrid_had_photons = Vec()
        self.h2l_lo = VecI()
        self.h2l_w = Vec()
        self.l2h_lo = VecI()
        self.l2h_w = Vec()
        self.n = List[Vec](capacity=P_COUNT)
        for _ in range(P_COUNT):
            self.n.append(Vec())
        self.update_energy_grids()
        self.clear_all_particle_arrays()

    def update_energy_grids(mut self):
        self.egrid_lep = zeros(self.nbins_lep)
        for i in range(self.nbins_lep):
            self.egrid_lep[i] = c_exp(Float64(i) * self.dlnE)
        self.egrid_pho = zeros(self.nbins_pho)
        for i in range(self.nbins_pho):
            self.egrid_pho[i] = c_exp(
                Float64(i - self.ind_min_energy_pho) * self.dlnE
            )
        self.egrid_had = zeros(self.nbins_had)
        for i in range(self.nbins_had):
            self.egrid_had[i] = c_exp(Float64(i) * self.dlnE)
        self.egrid_nu = zeros(self.nbins_nu)
        for i in range(self.nbins_nu):
            self.egrid_nu[i] = c_exp(
                Float64(i - self.ind_min_energy_nu) * self.dlnE
            )
        self.egrid_had_leptons = zeros(self.nbins_lep_hadgrid)
        for i in range(self.nbins_lep_hadgrid):
            self.egrid_had_leptons[i] = c_exp(
                Float64(i - self.ind_min_energy_lep_hadgrid) * self.dlnE
            )
        self.egrid_had_photons = zeros(self.nbins_pho_hadgrid)
        for i in range(self.nbins_pho_hadgrid):
            self.egrid_had_photons[i] = c_exp(
                Float64(i - self.ind_min_energy_pho_hadgrid) * self.dlnE
            )
        self._build_grid_maps()

    def _build_grid_maps(mut self):
        """The interpolation index and weight of every bin depend only on the
        grids, not on the values."""
        var n = max(self.nbins_pho, max(self.nbins_nu, self.nbins_lep))
        var origin_ratio = c_log(MASS_E / 1.0) / self.dlnE
        self.h2l_lo = zerosi(n)
        self.h2l_w = zeros(n)
        self.l2h_lo = zerosi(n)
        self.l2h_w = zeros(n)
        for i in range(n):
            for dir in range(2):
                var shifted = (
                    Float64(i + self.ind_min_energy_lep_hadgrid) + origin_ratio
                ) if dir == 0 else (
                    Float64(i - self.ind_min_energy_lep_hadgrid) - origin_ratio
                )
                var lo = Int(floor(shifted))
                var f = shifted - Float64(lo)
                var w = f + 0.5 * f * (1.0 - f) * self.dlnE
                if dir == 0:
                    self.h2l_lo[i] = lo
                    self.h2l_w[i] = w
                else:
                    self.l2h_lo[i] = lo
                    self.l2h_w[i] = w

    def update_energy_grid_parameters(
        mut self, Egamma_min_eV: Float64, Enu_min_eV: Float64, E_max_eV: Float64
    ):
        # AM3 truncates only. For some values of E_max, the lepton or hadron
        # grid then ended two bins below it. For some values of Egamma_min or
        # Enu_min, the photon or neutrino grid started one bin above its
        # minimum: AM3 bug.
        self.nbins_lep = self.eV_to_lep_grid_index(E_max_eV)
        if self.nbins_lep % 2 != 0:
            self.nbins_lep += 1
        self.nbins_had = self.eV_to_had_grid_index(E_max_eV)
        if self.nbins_had % 2 != 0:
            self.nbins_had += 1
        self.ind_min_energy_pho = Self._grid_index(
            c_log(MEC2_EV / Egamma_min_eV) / self.dlnE
        )
        self.nbins_pho = self.nbins_lep + self.ind_min_energy_pho
        if self.nbins_pho % 2 != 0:
            self.nbins_pho += 1
        self.ind_min_energy_nu = Self._grid_index(
            c_log(1e9 / Enu_min_eV) / self.dlnE
        )
        self.nbins_nu = self.nbins_had + self.ind_min_energy_nu
        if self.nbins_nu % 2 != 0:
            self.nbins_nu += 1
        self.nbins_lep_hadgrid = self.nbins_lep
        self.ind_min_energy_lep_hadgrid = 1 + Int(
            c_log(1e9 / MEC2_EV) / self.dlnE
        )
        self.nbins_pho_hadgrid = self.nbins_pho
        # As in the constructor, the photon grid is shifted like the lepton
        # grid, which grid_lep2had / grid_had2lep assume. AM3 took
        # 1 + int(ln(1e9 / Egamma_min) / dlnE), one more for 79 % of the values
        # of Egamma_min, every power of ten among them. So its photons on the
        # hadron grid were labeled e^-dlnE times their energy: AM3 bug.
        self.ind_min_energy_pho_hadgrid = (
            self.ind_min_energy_lep_hadgrid + self.ind_min_energy_pho
        )

    def nbins_of_particle(self, id: Int) -> Int:
        if id < P_HAD_END or (id >= P_EXT_HAD_BEGIN and id < P_EXT_HAD_END):
            return self.nbins_had
        if id >= P_EXT_LEP_BEGIN and id < P_EXT_LEP_END:
            return self.nbins_lep
        if id >= P_EXT_NU_BEGIN and id < P_EXT_NU_END:
            return self.nbins_nu
        elif id < P_NU_END:
            return self.nbins_nu
        elif id < P_LEP_END:
            return self.nbins_lep
        return self.nbins_pho

    def clear_all_particle_arrays(mut self):
        for id in range(P_COUNT):
            resize_zero(self.n[id], self.nbins_of_particle(id))

    # ---------------- grid shifts between hadronic / leptonic origin ----------
    @inline(.always)
    def grid_had2lep(self, i: Int, val: FPtr, nmax: Int) -> Float64:
        var lo = self.h2l_lo[i]
        if lo > 0 and lo < nmax - 1:
            var w = self.h2l_w[i]
            return (1.0 - w) * val[unsafe_offset=lo] + w * val[
                unsafe_offset=lo + 1
            ]
        return 0.0

    @inline(.always)
    def grid_lep2had(self, i: Int, val: FPtr, nmax: Int) -> Float64:
        """The C++ Grid_Lep2Had takes the whole spectrum by value: a copy of its
        620 values for each of the 620 bins it fills, every step."""
        var lo = self.l2h_lo[i]
        if lo > 0 and lo < nmax - 1:
            var w = self.l2h_w[i]
            return (1.0 - w) * val[unsafe_offset=lo] + w * val[
                unsafe_offset=lo + 1
            ]
        return 0.0

    @staticmethod
    def _grid_index(x: Float64) -> Int:
        """AM3 truncates only. The logarithm of a grid point's energy, divided
        by the spacing, comes out slightly below its index for 2 to 4 % of the
        points. For these points, truncation then gave the point below:
        AM3 bug."""
        var r = floor(x + 0.5)
        if abs(x - r) < 1e-9:
            return Int(r)
        return Int(x)

    def eV_to_lep_grid_index(self, E_eV: Float64) -> Int:
        return Self._grid_index(c_log(E_eV / MEC2_EV) / self.dlnE)

    def eV_to_photon_grid_index(self, E_eV: Float64) -> Int:
        return Self._grid_index(
            c_log(E_eV / MEC2_EV) / self.dlnE + Float64(self.ind_min_energy_pho)
        )

    def eV_to_had_grid_index(self, E_eV: Float64) -> Int:
        return Self._grid_index(c_log(E_eV / 1e9) / self.dlnE)

    def eV_to_neutrino_grid_index(self, E_eV: Float64) -> Int:
        return Self._grid_index(
            c_log(E_eV / 1e9) / self.dlnE + Float64(self.ind_min_energy_nu)
        )
