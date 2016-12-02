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

dt = 0.1     # [s]
dt_min = 0.1 # [s]
dt_max = 1.0  # [s]


# TS config

# options.setValue('-ts_type', 'bdf') # https://en.wikipedia.org/wiki/Backward_differentiation_formula
# options.setValue('-ts_bdf_order', 2) # https://en.wikipedia.org/wiki/Backward_differentiation_formula
# options.setValue('-ts_bdf_adapt', None)
# NOTE for bdf max clip is 2, see bdf.c line 379: high = PetscMin(high,2.0);
# options.setValue('-ts_adapt_basic_clip', (0.11, 2.0)) # Admissible decrease/increase factor in step size (TSAdaptBasicSetClip)

# options.setValue('-ts_type', 'python') # https://en.wikipedia.org/wiki/Backward_differentiation_formula
# options.setValue('-ts_type', 'pseudo') # https://en.wikipedia.org/wiki/Backward_differentiation_formula
options.setValue('-ts_type', 'theta') # https://en.wikipedia.org/wiki/Backward_differentiation_formula
options.setValue('-ts_adapt_type', 'basic') # basic or none
options.setValue('-ts_theta_adapt', None)
# options.setValue('-ts_theta_theta', 0.5)
# options.setValue('-ts_theta_initial_guess_extrapolate', None)
# options.setValue('-ts_theta_endpoint', None) # activate this for Crank-Nicholson
# options.setValue('-ts_theta_theta', 0.8)

options.setValue('-ts_rtol', 0.01)
options.setValue('-ts_atol', 0.01)
options.setValue('-ts_adapt_dt_min', dt_min)
options.setValue('-ts_adapt_dt_max', dt_max)
options.setValue('-ts_exact_final_time', 'matchstep') # http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/TS/TSSetExactFinalTime.html
options.setValue('-ts_adapt_scale_solve_failed', 0.5) # Scale step by this factor if solve fails ()
options.setValue('-ts_adapt_basic_clip', (0.1, 10.0)) # Admissible decrease/increase factor in step size (TSAdaptBasicSetClip)
options.setValue('-ts_adapt_basic_safety', 0.9) # Safety factor relative to target error ()
options.setValue('-ts_adapt_basic_reject_safety', 0.5) # Extra safety factor to apply if the last step was rejected ()
options.setValue('-ts_adapt_basic_always_accept', None) # Always accept the step regardless of whether local truncation error meets goal ()
# options.setValue('-ts_error_if_step_fails', False) # Always accept the step regardless of whether local truncation error meets goal ()
# options.setValue('-ts_adapt_wnormtype', 'infinity')
# options.setValue('-ts_max_steps', 10000000)
options.setValue('-ts_monitor', None)
# options.setValue('-ts_adjoint_solve', True)

# options.setValue('-ts_monitor_lg_timestep', None)



# options.setValue('-ts_pseudo_increment', 1.1) # - ratio of increase dt
# options.setValue('-ts_pseudo_increment_dt_from_initial_dt', None) # Increase dt as a ratio from original dt
# options.setValue('-ts_pseudo_fatol', 0.0001) # stop iterating when the function norm is less than atol
# options.setValue('-ts_pseudo_frtol', 0.0001) # stop iterating when the function norm divided by the initial function norm is less than rtol
# options.setValue('-ts_pseudo_max_dt', dt_max) 

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

# options.setValue('-sub_0_snes_monitor_short', None)
# options.setValue('-sub_0_snes_converged_reason', None)
# options.setValue('-sub_1_snes_monitor_short', None)
# options.setValue('-sub_1_snes_converged_reason', None)

# options.setValue('-sub_0_fieldsplit_0_ksp_monitor_short', None)
# options.setValue('-sub_0_fieldsplit_1_ksp_monitor_short', None)
# options.setValue('-sub_1_fieldsplit_0_ksp_monitor_short', None)
# options.setValue('-sub_1_fieldsplit_1_ksp_monitor_short', None)
# options.setValue('-sub_0_fieldsplit_0_ksp_monitor_short', None)
# options.setValue('-sub_0_fieldsplit_1_ksp_monitor_short', None)
# options.setValue('-sub_1_fieldsplit_0_ksp_monitor_short', None)
# options.setValue('-sub_1_fieldsplit_1_ksp_monitor_short', None)
# 
# options.setValue('-sub_0_fieldsplit_0_ksp_converged_reason', None)
# options.setValue('-sub_0_fieldsplit_1_ksp_converged_reason', None)
# options.setValue('-sub_1_fieldsplit_0_ksp_converged_reason', None)
# options.setValue('-sub_1_fieldsplit_1_ksp_converged_reason', None)
# options.setValue('-sub_0_fieldsplit_0_ksp_converged_reason', None)
# options.setValue('-sub_0_fieldsplit_1_ksp_converged_reason', None)
# options.setValue('-sub_1_fieldsplit_0_ksp_converged_reason', None)
# options.setValue('-sub_1_fieldsplit_1_ksp_converged_reason', None)

# options.setValue('-snes_convergence_test', 'skip')
# options.setValue('-snes_max_it', 10)
# options.setValue('-snes_linesearch_type', 'basic')
# options.setValue('-snes_function_type', 'preconditioned')
# options.setValue('-snes_type', 'nasm')
# 
# options.setValue('snes_nasm_type', 'basic') # <BASIC> (choose one of) NONE RESTRICT INTERPOLATE BASIC ()
# options.setValue('snes_nasm_damping', 1 ) #The new solution is obtained as old solution plus dmp times (sum of the solutions on the subdomains) (SNESNASMSetDamping)
# options.setValue('snes_nasm_sub_view', False) # <FALSE> Print detailed information for every processor when using -snes_view ()
# options.setValue('snes_nasm_finaljacobian', False) #: <FALSE> Compute the global jacobian of the final iterate (for ASPIN) ()
# options.setValue('snes_nasm_finaljacobian_type', 'finalouter') #<FINALOUTER> (choose one of) FINALOUTER FINALINNER INITIAL ()
# options.setValue('snes_nasm_log', True)   #: <TRUE> Log times for subSNES solves and restriction ()

options.setValue('-snes_stol', 1e-8)
options.setValue('-snes_rtol', 1e-8)
options.setValue('-snes_atol', 1e-8)
# options.setValue('-snes_type', 'newtonls')
# options.setValue('-snes_linesearch_type', 'basic')
# options.setValue('-npc_snes_type', 'ngs')
# options.setValue('-npc_snes_type', 'fas')

# options.setValue('pc_factor_shift_type', 'NONZERO')
# options.setValue('pc_factor_shift_amount', 1e-12)
###############
############
# NEW STUFF!!!
# options.setValue('-npc_snes_type', 'multiblock')
# options.setValue('-snes_type', 'multiblock')
# options.setValue('-snes_multiblock_block_size', 5) # Blocksize that defines number of fields
# options.setValue('-snes_multiblock_0_fields', '0,1')
# options.setValue('-snes_multiblock_1_fields', '2,3,4')

# options.setValue('-snes_multiblock_type', '') # Type of composition PCFieldSplitSetType
# -snes_multiblock_default
# -snes_multiblock_detect_saddle_point
#     options.setValue(snes + 'pc_type', 'fieldsplit')  
#     options.setValue(snes + 'pc_fieldsplit_type', 'schur')  
#     options.setValue(snes + 'pc_fieldsplit_schur_fact_type', 'lower')   
###############
############

# options.setValue('-snes_vi_zero_tolerance', 1e-8) # Tolerance for considering x[] value to be on a bound (None)
# options.setValue('-snes_vi_monitor', True) # Monitor all non-active variables (SNESMonitorResidual)
# options.setValue('-snes_vi_monitor_residual', True) # Monitor residual all non-active variables; using zero for active constraints (SNESMonitorVIResidual)
   
options.setValue('-snes_type', 'composite')
options.setValue('-snes_composite_type', 'additiveoptimal')

# newtonls newtontr test nrichardson ksponly vinewtonrsls 
# vinewtonssls ngmres qn shell ngs ncg fas ms nasm anderson 
# aspin composite python (SNESSetType)
# options.setValue('-snes_composite_sneses', 'ksponly,newtonls,nrichardson')
# options.setValue('-snes_composite_sneses', 'vinewtonrsls,newtonls')
# options.setValue('-snes_composite_sneses', 'fas,vinewtonrsls')
# options.setValue('-snes_composite_sneses', 'newtonls')
# options.setValue('-snes_composite_sneses', 'ksponly,newtonls')
options.setValue('-snes_composite_sneses', 'nrichardson,newtonls')
# options.setValue('-snes_composite_sneses', 'ngs,newtonls')
# options.setValue('-snes_composite_damping', 0.5) # Damping of the additive composite solvers (SNESCompositeSetDamping)
# options.setValue('-snes_composite_stol', 0.1) # Step tolerance for restart on the additive composite solvers ()
# options.setValue('-snes_composite_stol', 1.1) # Residual tolerance for the additive composite solvers ()


# options.setValue('-snes_composite_sneses', 'fas,newtonls')
# options.setValue('-snes_composite_sneses', 'ngmres,newtonls')
 
# options.setValue('da_refine_x', 1)
# options.setValue('sub_0_fas_levels_snes_type', 'gs')
# options.setValue('sub_0_fas_levels_snes_max_it', 1)
# options.setValue('sub_0_fas_coarse_snes_type', 'ngs')
# options.setValue('sub_0_fas_coarse_snes_max_it', 1)
# options.setValue('sub_0_fas_coarse_snes_grid_sequence', 0)
# options.setValue('sub_0_fas_coarse_snes_linesearch_type', 'basic')
# options.setValue('sub_0_fas_coarse_snes_type', 'newtonls')

# options.setValue('sub_0_snes_linesearch_type', 'bt')
# options.setValue('sub_1_snes_linesearch_type', 'basic')
# options.setValue('sub_1_npc_snes_type', 'ngs')
# options.setValue('sub_1_pc_type', 'mg')
# options.setValue('sub_0_snes_stol', 1e-50)
# options.setValue('sub_0_snes_rtol', 1e-50)
# options.setValue('sub_0_snes_atol', 1e-6)
# options.setValue('sub_1_snes_stol', 1e-50)
# options.setValue('sub_1_snes_rtol', 1e-50)
# options.setValue('sub_1_snes_atol', 1e-6)
# options.setValue('sub_0_snes_convergence_test', 'skip')
# options.setValue('sub_0_snes_max_it', 10)




# # for field split solver
# # for snes in ['sub_0_', 'sub_1_']: # for snes == composite 
# # for snes in ['sub_0_']: # for snes == composite 
# # for snes in ['sub_']: # for snes == composite 
# for snes in ['']:
# # for snes in ['', 'npc_']:
# # for snes in ['npc_']:
#     # ksp config
# #     options.setValue(snes + 'ksp_type', 'fgmres')  
#               
#     # pc config
#     # For direct solver
#     # options.setValue(snes + 'pc_type', 'lu')
#     # options.setValue(snes + 'ksp_type', 'preonly')
# #     options.setValue(snes + 'pc_factor_shift_type', 'NONZERO')
# #     options.setValue(snes + 'pc_factor_shift_amount', 1e-12)
#                   
#     options.setValue(snes + 'pc_type', 'fieldsplit')  
# #     options.setValue(snes + 'pc_fieldsplit_type', 'special')  
# #     -pc_fieldsplit_off_diag_use_amat: <FALSE> Use Amat (not Pmat) to extract off-diagonal fieldsplit blocks (PCFieldSplitSetOffDiagUseAmat)
# #   -pc_fieldsplit_type <MULTIPLICATIVE> (choose one of) ADDITIVE MULTIPLICATIVE SYMMETRIC_MULTIPLICATIVE SPECIAL SCHUR (PCFieldSplitSetType)
#     options.setValue(snes + 'pc_fieldsplit_type', 'schur')  
#     options.setValue(snes + 'pc_fieldsplit_schur_fact_type', 'lower')   
#     options.setValue(snes + 'pc_fieldsplit_block_size', 5)   
#     options.setValue(snes + 'pc_fieldsplit_0_fields', '0,1')   
#     options.setValue(snes + 'pc_fieldsplit_1_fields', '2,3,4')   
#                   
# #     options.setValue(snes + 'fieldsplit_0_ksp_rtol', 1000)
# #     options.setValue(snes + 'fieldsplit_0_ksp_atol', 1000)
# #     options.setValue(snes + 'fieldsplit_0_ksp_convergence_test', 'skip')
# #     options.setValue(snes + 'fieldsplit_1_ksp_convergence_test', 'skip')
#       
#     options.setValue(snes + 'fieldsplit_0_ksp_type', 'gmres')
# #     options.setValue(snes + 'fieldsplit_0_pc_type', 'bjacobi')
# #     options.setValue(snes + 'fieldsplit_0_pc_bjacobi_blocks', 2)
# #     options.setValue(snes + 'fieldsplit_0_sub_pc_type', 'jacobi')                 
# #     options.setValue(snes + 'fieldsplit_0_sub_pc_bjacobi_blocks', 2)
# #                   
#     options.setValue(snes + 'fieldsplit_1_ksp_type', 'gmres')    
# #     options.setValue(snes + 'fieldsplit_1_pc_type', 'lsc')
# #     options.setValue(snes + 'fieldsplit_1_lsc_pc_type', 'ml')
# #     options.setValue(snes + 'fieldsplit_1_ksp_type', 'preonly')    
# #     options.setValue(snes + 'fieldsplit_1_pc_type', 'lu')
# #     options.setValue(snes + 'fieldsplit_1_pc_factor_shift_type', 'NONZERO')
# #     options.setValue(snes + 'fieldsplit_1_pc_factor_shift_amount', 1e-6)
# #     options.setValue(snes + 'fieldsplit_1_ksp_type', 'preonly') 
# #     options.setValue(snes + 'fieldsplit_1_pc_type', 'jacobi') 
# #     options.setValue(snes + 'fieldsplit_1_pc_jacobi_type', 'diagonal') # (choose one of) DIAGONAL ROWMAX ROWSUM (PCJacobiSetType)
# #                   
# #     options.setValue(snes + 'fieldsplit_1_inner_ksp_type', 'preonly')
# #     options.setValue(snes + 'fieldsplit_1_inner_pc_type', 'jacobi')
# #                  
# #     options.setValue(snes + 'fieldsplit_1_upper_ksp_type', 'preonly')
# #     options.setValue(snes + 'fieldsplit_1_upper_pc_type', 'jacobi')
#     
# #     options.setValue(snes + 'fieldsplit_1_upper_pc_jacobi_type', 'diagonal') # (choose one of) DIAGONAL ROWMAX ROWSUM (PCJacobiSetType)
# #                   
# #     options.setValue(snes + 'fieldsplit_1_mat_schur_complement_ainv_type', 'lump')   
# #     options.setValue(snes + 'pc_fieldsplit_schur_precondition', 'selfp')   
# #     options.setValue(snes + 'pc_fieldsplit_schur_precondition', 'full')   
#     options.setValue(snes + 'pc_fieldsplit_schur_precondition', 'a11')   
# #     options.setValue(snes + 'pc_fieldsplit_detect_saddle_point', None)
# #     options.setValue(snes + 'pc_fieldsplit_default', None)
     
# time_intervals = [1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 0.2, 10]
# time_intervals = [1e-4, 1e-3]
# time_intervals = [1.3133]
time_intervals = np.linspace(0,250, num=250)
# time_intervals = np.linspace(0,2, num=100)
# time_intervals = np.concatenate((np.logspace(-4, -0.1, num=200), np.linspace(1,250))) 

npipes = 1
nx = 100

nphases = 2
dof = nphases * 2 + 1

pipe_length = 1000.0 # [m]

f, axarr = plt.subplots(4, sharex=True)
axarr[0].set_title('Results')

initial_solution = np.zeros((nx,dof))
initial_solution[:,0:nphases] = 0.1 # Velocity

αG = 0.5
initial_solution[:,2] = αG # vol frac
initial_solution[:,3] = 1-αG # vol frac

initial_solution[:,-1] = 1.0#np.linspace(2,1.01,num=nx)  # Pressure
initial_time = 0.0

sols = []
for i, final_time in enumerate(time_intervals):

    sol, final_dt = transient_pipe_flow_1D(
        npipes, nx, dof, nphases,
        pipe_length,
        initial_time,
        final_time,
        dt,
        dt_min,
        dt_max,
        initial_solution.flatten(),
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
#     axarr[1].set_ylim(0, 4)
#     axarr[3].set_ylim(0.99, 1.01)
#     plt.show()
    
    plt.draw()
    plt.pause(0.0001)

