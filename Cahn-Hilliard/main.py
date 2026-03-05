import os
from pathlib import Path

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np

import ufl
from basix.ufl import element, mixed_element
from dolfinx import default_real_type, log, plot
from dolfinx.fem import Function, functionspace
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, create_unit_square
from dolfinx.fem import assemble_scalar, form

try:
    import pyvista as pv
    import pyvistaqt as pvqt

    have_pyvista = True
except ModuleNotFoundError:
    print("pyvista and pyvistaqt are required to visualise the solution")
    have_pyvista = False


# Save all logging to file
log.set_output_file("log.txt")

#parameters
lmbda = 1.0e-02  # surface parameter
theta = 0.5  # time stepping family, e.g. theta=1 -> backward Euler, theta=0.5 -> Crank-Nicholson
t = 0.0  # Current time
N_snapshots = 50        # <-- you choose this
T = 0.01                 # total simulation time (adjust to see ripening)
dt = 1e-5               # consider 1e-5 to 5e-6 to start

# Create mesh and function space
msh = create_unit_square(MPI.COMM_WORLD, 96, 96, CellType.triangle)
P1 = element("Lagrange", msh.basix_cell(), 1)
ME = functionspace(msh, mixed_element([P1, P1]))

# Define trial and test functions
q, v = ufl.TestFunctions(ME)

u = Function(ME)  # current solution
u0 = Function(ME)  # solution from previous converged step

# Split mixed functions
c, mu = ufl.split(u)
c0, mu0 = ufl.split(u0)

# Interpolate initial condition or restart from old output
out_folder = Path("demo_ch")
out_folder.mkdir(parents=True, exist_ok=True)
restart_file = out_folder / "checkpoint.xdmf"

if restart_file.exists():
    print("Restarting from checkpoint")

    with XDMFFile(MPI.COMM_WORLD, restart_file, "r") as f:
        f.read_function(u.sub(0))
        f.read_function(u.sub(1))

    u0.x.array[:] = u.x.array

else:
    print("Starting from random initial condition")

    rng = np.random.default_rng(42)
    u.sub(0).interpolate(lambda x: 0.63 + 0.02 * (0.5 - rng.random(x.shape[1]))) # adding oscillations
    u.x.scatter_forward()

    u0.x.array[:] = u.x.array

# Compute the chemical potential df/dc
c = ufl.variable(c)
f = c**2 * (1 - c) ** 2 # double well potential. Well at c=0, and c=1.
dfdc = ufl.diff(f, c)

mu_mid = (1.0 - theta) * mu0 + theta * mu  # mu_(n+theta)

# Weak statement of the equations
F0 = (
    ufl.inner(c, q) * ufl.dx
    - ufl.inner(c0, q) * ufl.dx
    + dt * ufl.inner(ufl.grad(mu_mid), ufl.grad(q)) * ufl.dx
)
F1 = (
    ufl.inner(mu, v) * ufl.dx
    - ufl.inner(dfdc, v) * ufl.dx
    - lmbda * ufl.inner(ufl.grad(c), ufl.grad(v)) * ufl.dx
)
F = F0 + F1

# Diagnostics: mass and free energy
mass_form = form(c * ufl.dx)

energy_density = f + 0.5 * lmbda * ufl.inner(ufl.grad(c), ufl.grad(c))
energy_form = form(energy_density * ufl.dx)


use_superlu = PETSc.IntType == np.int64  # heuristic to decide whether the current PETSc installation was built for 64-bit integers.
sys = PETSc.Sys()  # Creates a handle to the PETSc system object
if sys.hasExternalPackage("mumps") and not use_superlu:
    linear_solver = "mumps"
elif sys.hasExternalPackage("superlu_dist"):
    linear_solver = "superlu_dist"
else:
    linear_solver = "petsc"

petsc_options = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "none",
    "snes_stol": np.sqrt(np.finfo(default_real_type).eps) * 1e-2,
    "snes_atol": 0,
    "snes_rtol": 0,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": linear_solver,
    "snes_monitor": None,
}

problem = NonlinearProblem(F, u)
solver = NewtonSolver(MPI.COMM_WORLD, problem)
# Configure solver
solver.rtol = 1e-8
solver.atol = 1e-8
solver.max_it = 10000
solver.error_on_nonconvergence = True

solver.report = True

# Creating output files:
file = XDMFFile(MPI.COMM_WORLD, out_folder / "output.xdmf", "w")  # Output file
file.write_mesh(msh)

# Save initial state at t=0
c_fun = u.sub(0)
file.write_function(c_fun, t)

# Checkpoint file for restart (stores the full mixed solution u = (c, mu))
checkpoint_file = XDMFFile(MPI.COMM_WORLD, out_folder / "checkpoint.xdmf", "w")
checkpoint_file.write_mesh(msh)

V0, dofs = ME.sub(0).collapse()

#For pyvista.
if have_pyvista:
    # Create a VTK 'mesh' with 'nodes' at the function dofs
    topology, cell_types, x = plot.vtk_mesh(V0)
    grid = pv.UnstructuredGrid(topology, cell_types, x)

    # Set output data
    grid.point_data["c"] = u.x.array[dofs].real
    grid.set_active_scalars("c")

    p = pvqt.BackgroundPlotter(title="concentration", auto_update=True)
    p.add_mesh(grid, clim=[0, 1])
    p.view_xy(negative=True)
    p.add_text(f"time: {t}", font_size=12, name="timelabel")

c = u.sub(0)
u0.x.array[:] = u.x.array
step = 0

#computing initial diagnostic.
mass0 = msh.comm.allreduce(assemble_scalar(mass_form), op=MPI.SUM)
energy0 = msh.comm.allreduce(assemble_scalar(energy_form), op=MPI.SUM)
print(f"Initial mass = {mass0:.8e}")
print(f"Initial energy = {energy0:.8e}")

# Snapshot schedule: N times from 0 to T inclusive
snapshot_times = np.linspace(0.0, T, N_snapshots)
snap_id = 1  # we already wrote snapshot at t=0, so next is snapshot_times[1]

while t < T:
    t += dt
    n, converged = solver.solve(u)
    assert converged
    # Diagnostics
    mass = msh.comm.allreduce(assemble_scalar(mass_form), op=MPI.SUM)
    energy = msh.comm.allreduce(assemble_scalar(energy_form), op=MPI.SUM)
    gradc2_form = form(ufl.inner(ufl.grad(c), ufl.grad(c)) * ufl.dx) # detext \int \nabla c^2. Dec with coarsening.

    print(
        f"Step {step} | Newton iterations {n} | "
        f"mass = {mass:.8e} | energy = {energy:.8e} | gradc2 = {msh.comm.allreduce(assemble_scalar(gradc2_form), op=MPI.SUM):.8e}"
    )

    u0.x.array[:] = u.x.array
    file.write_function(c, t)          # visualization (only c)
    if step > 0 and step % 50 == 0:
       print(f"Checkpoint saved at step {step}, time {t:.3e}")
       checkpoint_file.write_function(u.sub(0), t)   # concentration
       checkpoint_file.write_function(u.sub(1), t)   # chemical potential

    step += 1

    # Write snapshots whenever we pass the next target time
    while snap_id < N_snapshots and t >= snapshot_times[snap_id] - 1e-14:
        file.write_function(c_fun, snapshot_times[snap_id])
        snap_id += 1

    # Update the plot window
    if have_pyvista:
        p.add_text(f"time: {t:.2e}", font_size=12, name="timelabel")
        grid.point_data["c"] = u.x.array[dofs].real
        p.app.processEvents()

file.close()
checkpoint_file.close()

if have_pyvista:
    grid.point_data["c"] = u.x.array[dofs].real
    screenshot = out_folder / "ch.png" if pv.OFF_SCREEN else None
    pv.plot(grid, show_edges=True, screenshot=screenshot)
