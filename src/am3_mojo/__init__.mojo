"""am3_mojo: the AM3 (Astrophysical Multi-Messenger Modeling) code in Mojo.

utils/       numerics: libm wrappers, constants, buffers, linear algebra,
             interpolation, special functions, parallel helpers
model/       run parameters and the energy grids / particle arrays
solver/      the implicit Chang-Cooper PDE solver
physics/     radiation and interaction modules
simulation/  the time-step orchestration (SimulationManager)
api/         the AM3 object exposed to Python
"""
