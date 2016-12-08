import sys
import numpy as np
import petsc4py
from petsc4py import PETSc
from matplotlib import pyplot as plt
from flow import transient_pipe_flow_1D

np.set_printoptions(precision=3, linewidth=300)
petsc4py.init(sys.argv)
options = PETSc.Options()
options.clear()

options.setValue('-mat_fd_type', 'ds')
options.setValue('-mat_fd_coloring_err', 1e-3)
options.setValue('-mat_fd_coloring_umin', 1e-5)

# options.setValue('-mat_view', 'draw')
# options.setValue('-draw_pause', 5000)
#options.setValue('-is_coloring_view', '')
# options.setValue('-help', None)

options.setValue('-snes_monitor_short', None)
options.setValue('-snes_converged_reason', None)

# options.setValue('-snes_convergence_test', 'skip')
options.setValue('-snes_max_it', 10)

options.setValue('-snes_stol', 1e-50)
options.setValue('-snes_rtol', 1e-50)
options.setValue('-snes_atol', 1e-5)

# pc config For direct solver
options.setValue('pc_type', 'lu')
options.setValue('ksp_type', 'preonly')
options.setValue('pc_factor_shift_type', 'NONZERO')
# options.setValue('pc_factor_shift_amount', 1e-14)
# options.setValue('pc_factor_nonzeros_along_diagonal', None)

# Normal volume fractions, pure newton is ok
αG = 0.1
options.setValue('-snes_type', 'newtonls')
# options.setValue('-snes_type', 'vinewtonrsls')
# options.setValue('-npc_snes_type', 'ngs')

# options.setValue('-snes_type', 'composite')
# options.setValue('-snes_composite_type', 'additiveoptimal')
# # options.setValue('-snes_composite_sneses', 'fas,newtonls')
# options.setValue('-snes_composite_sneses', 'newtonls,newtonls')
# options.setValue('-snes_composite_damping', (0.,0.8)) # Damping of the additive composite solvers (SNESCompositeSetDamping)
# options.setValue('-sub_1_snes_linesearch_type', 'basic')
# # options.setValue('-sub_1_npc_snes_type', 'ngs')

options.setValue('-snes_linesearch_type', 'basic')
# time_intervals = np.linspace(0.1, 25, num=250) # [s]
time_intervals = np.linspace(0.1, 200, num=2500) # [s]
# time_intervals = [20.0]
dt = 0.001     # [s]
dt_min = 0.00001 # [s]
dt_max = (25-0.1)/250  # [s]


nx = 1000
npipes = 1
nphases = 2
dof = nphases * 2 + 1

    
Ppresc  = 1.0 # [bar]
Mpresc = [0.02, 3.0] # [kg/s]    

diameter = 0.1 # [m]
pipe_length = 100.0 # [m]

initial_solution = np.zeros((nx,dof))
initial_solution[:,0:nphases] = 1e-8 # Velocity
initial_solution[:,2] = αG # vol frac
initial_solution[:,3] = 1-αG # vol frac
initial_solution[:,-1] = 1.0  # Pressure
initial_time = 0.0

f, axarr = plt.subplots(4, sharex=True, figsize=(12,8))
sols = []
for i, final_time in enumerate(time_intervals):

    sol, final_dt = transient_pipe_flow_1D(
        npipes, nx, dof, nphases,
        pipe_length, diameter,
        initial_time,
        final_time,
        dt,
        dt_min,
        dt_max,
        initial_solution.flatten(),
        Mpresc,
        Ppresc,
        impl_python=True
        )
    
    SOL = sol[...].reshape(nx, dof)

    U = SOL[:, 0:nphases]
    α = SOL[:, nphases:-1]
    P = SOL[:, -1]
    
    initial_solution[:,0:nphases] = U # Velocity
    initial_solution[:,nphases:-1] = α # vol frac
    initial_solution[:,-1] = P   # Pressure

    initial_time = final_time
    dt = final_dt
             
    sols.append((U, P))
    
    dx = pipe_length / (nx - 1)
    x = np.linspace(0, npipes*pipe_length, npipes*nx, endpoint=True) + 0.5*dx
    xx = np.concatenate((x[:-1], [pipe_length]))


    UU = U
    PP = np.concatenate((P[:-1], [1.0]))
    αα = α
    
    axarr[0].set_title('Results')
    axarr[0].cla()
    axarr[1].cla()
    axarr[2].cla()
    axarr[3].cla()
    axarr[0].plot(xx, αα, '-', linewidth=2)
    axarr[1].plot(xx, UU, '-', linewidth=2)
    axarr[2].plot(xx, αα*UU, '-', linewidth=2)
    axarr[3].plot(xx, PP, 'r-', linewidth=2)
    plt.xlim(0, pipe_length)
    axarr[0].set_ylim(0, 2)
    plt.draw()
    plt.pause(0.0001)
    
# (0.002, 0.3, [0.13499, 0.10906, 0.0555], [102391.457, 102369.307, 102269.131],
print('pressure', PP[[5, 10, 20]])
print('vol frac', αα[[5, 10, 20]])