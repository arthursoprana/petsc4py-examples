import sys
import numpy as np
import petsc4py
from petsc4py import PETSc
from matplotlib import pyplot as plt
from flow import transient_pipe_flow_1D

petsc4py.init(sys.argv)
options = PETSc.Options()
options.clear()

# TS config
options.setValue('-ts_type', 'theta') # https://en.wikipedia.org/wiki/Backward_differentiation_formula
options.setValue('-ts_adapt_type', 'basic') # basic or none
options.setValue('-ts_theta_adapt', None)
options.setValue('-ts_rtol', 0.01)
options.setValue('-ts_atol', 0.01)
options.setValue('-ts_exact_final_time', 'matchstep') # http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/TS/TSSetExactFinalTime.html
options.setValue('-ts_adapt_scale_solve_failed', 0.5) # Scale step by this factor if solve fails ()
options.setValue('-ts_adapt_basic_clip', (0.1, 10)) # Admissible decrease/increase factor in step size (TSAdaptBasicSetClip)
options.setValue('-ts_adapt_basic_safety', 0.9) # Safety factor relative to target error ()
options.setValue('-ts_adapt_basic_reject_safety', 0.5) # Extra safety factor to apply if the last step was rejected ()
# options.setValue('-ts_adapt_basic_always_accept', None) # Always accept the step regardless of whether local truncation error meets goal ()
# options.setValue('-ts_error_if_step_fails', False) # Always accept the step regardless of whether local truncation error meets goal ()
options.setValue('-ts_max_steps', 10000000)
options.setValue('-ts_monitor', None)
# options.setValue('-ts_adjoint_solve', True)


dense = False
if dense:
    options.setValue('-dmcomposite_dense_jacobian', None)
else:
    options.delValue('-dmcomposite_dense_jacobian')

options.setValue('-ts_fd_color', None)

options.setValue('-mat_fd_type', 'ds')
# options.setValue('-mat_fd_coloring_err', 1e-2)
# options.setValue('-mat_fd_coloring_umin', 1e-3)

options.setValue('-ts_max_snes_failures', -1)
options.setValue('-ts_max_reject', -1)

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

# Normal volume fractions, pure newton is ok
αG = 0.01
options.setValue('-snes_type', 'newtonls')
# options.setValue('-snes_linesearch_type', 'basic')
time_intervals = np.linspace(0.1, 250, num=250) # [s]
dt = 0.001     # [s]
dt_min = 0.0001 # [s]
dt_max = 10.0  # [s]

# Lower volume fractions, need an initial low timestep
# αG = 0.001 # Lower than that == bad idea
# options.setValue('-snes_type', 'composite')
# options.setValue('-snes_composite_type', 'additive')
# options.setValue('-snes_composite_sneses', 'newtonls,newtonls')
# options.setValue('-snes_composite_damping', (0,1)) # Damping of the additive composite solvers (SNESCompositeSetDamping)
# options.setValue('-sub_1_snes_linesearch_type', 'basic')
# options.setValue('-sub_1_npc_snes_type', 'ngs')
# # time_intervals = np.logspace(-3, 0, num=100) # [s]
# time_intervals = [20] # [s]
# dt = 0.001     # [s]
# dt_min = 0.001 # [s]
# dt_max = 10.0  # [s]


# Lower volume fractions, need an initial low timestep
# αG = 1e-5 # Lower than that == bad idea
# options.setValue('-snes_type', 'composite')
# options.setValue('-snes_composite_type', 'additive')
# options.setValue('-snes_composite_sneses', 'fas,newtonls')
# options.setValue('-snes_composite_damping', (1,1)) # Damping of the additive composite solvers (SNESCompositeSetDamping)
# # options.setValue('-sub_1_snes_linesearch_type', 'basic')
# # options.setValue('-sub_1_npc_snes_type', 'ngs')
# time_intervals = np.logspace(-3, 0, num=100) # [s]
# dt = 0.001     # [s]
# dt_min = 0.001 # [s]
# dt_max = 10.0  # [s]


nx = 50
npipes = 1
nphases = 2
dof = nphases * 2 + 1

    
Ppresc  = 1.0 # [bar]
Mpresc = [0.002, 0.3] # [kg/s]    

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
    axarr[0].plot(xx, αα, '.-')
    axarr[1].plot(xx, UU, '.-')
    axarr[2].plot(xx, αα*UU, '.-')
    axarr[3].plot(xx, PP, 'r.-')
    plt.xlim(0, pipe_length)
    axarr[0].set_ylim(0, 1)
    plt.draw()
    plt.pause(0.0001)

# [102391.457, 102369.307, 102269.131]
# [0.13499, 0.10906, 0.0555]
print('pressure', PP[[5, 10, 20]])
print('vol frac', αα[[5, 10, 20]])