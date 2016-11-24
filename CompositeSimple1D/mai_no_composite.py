import sys
import numpy as np
import petsc4py
from petsc4py import PETSc
from matplotlib import pyplot as plt
import CompositeSimple1D



petsc4py.init(sys.argv)

def colebrook_white_explicit_friction_factor(reynolds, volume_fraction, diameter, absolute_rugosity):
    '''
    From Biberg 2008 - APPENDIX E: AN EXPLICIT APPROXIMATION FOR THE
                       COLEBROOK-WHITE FORMULA
    '''
    Re = np.abs(reynolds + 1.0e-15)
    ks = absolute_rugosity
    D  = diameter
    
    invf_haland = -1.8 * np.log10(6.9 / Re + (ks / (3.7 * D)) ** 1.11)
    invf_haland = np.where(invf_haland < 0.0, 0.0, invf_haland)
    
    invf0 = invf_haland
    
    # There is a 1/lambda_s (1/f_s) in the formula, but Biberg did not define this 
    # variable, so we'll use 1/f_0 (1/lambda_0)
    invfs = invf0
    
    term = (2.51 * invf0 / Re + ks / (3.7 * D))
    
    invf = ((5.02 * invfs / Re) - 4.6 * term * np.log10(term)) / ((5.02 / Re) + 2.3 * term)
    f_w_turbulent = 1.0 / invf ** 2
    
    # Avoid numpy warnings
    f_w_turbulent[np.where(np.isnan(f_w_turbulent))] = 0.0
    
    f_w_laminar = 64.0 / Re
    
    f_w = np.where(Re < 100.0, f_w_laminar, np.maximum(f_w_laminar, f_w_turbulent))
    
    fanning_factor = 0.25
    return fanning_factor * f_w


def ideal_gas_density_model(P, deriv=False):
    Z = 1.0
    R = 8.315
    T = 300.0
    M = 16.0e-3
      
    if deriv:
        return 0 * P + 1 / (Z * (R/M) * T) # 1/a**2
    else:
        return P * M / (Z * R * T)
    
# def ideal_gas_density_model(P, deriv=False):
#      
#     if deriv:
#         return 0 * P
#     else:
#         return 1000 + 0*P

class Heat(object):
    def __init__(self, dm, nx, dof, pipe_length):

        self.dm  = dm        
        self.L   = pipe_length
        self.nx  = nx
        self.dof = dof
        
    def evalFunction(self, ts, t, x, xdot, f):
        dm  = self.dm
        L   = self.L
        dof = self.dof
        nx  = self.nx
        
        with dm.getAccess(x, locs=None) as Xs:
            with dm.getAccess(xdot, locs=None) as X_ts:
                with dm.getAccess(f, locs=None) as Fs:
                    
                    for X, X_t, F in zip(Xs[:], X_ts[:], Fs[:]):   

                        udot = X_t.getArray(readonly=True)
                        u    = X.getArray(readonly=True)
                        
                        udot = udot.reshape(nx, dof)
                        dtP, dtU = udot[:, 1], udot[:, 0]
                        
                        u = u.reshape(nx, dof)
                        P, U = u[:, 1], u[:, 0] 
                        
                        D = 0.1 # [m]                        
                        ρ = ideal_gas_density_model(P*1e5)
                        c = ideal_gas_density_model(P, deriv=True)

                        Ppresc = 1.0 # [bar]
                        Upresc = 20.0 # [m/s]
                        
                        A = 0.25 * np.pi * D ** 2 # [m]
                        dx = L / F.size
                        ΔV = A * dx
                        
                        ρf = 0.5 * (ρ[:-1] + ρ[1:])
                        cf = 0.5 * (c[:-1] + c[1:])
                        ρf = np.concatenate(([ρf[0]], ρf))
                        cf = np.concatenate(([cf[0]], cf))
                        
                        Re = ρ * np.abs(U) * D / 1e-3
                        
                        fw = colebrook_white_explicit_friction_factor(Re, None, D, absolute_rugosity=1e-5)
                        τw = 0.5 * fw * ρf * np.abs(U) * U  
                        Sw = np.pi * D
                        
                        ff = np.zeros_like(u)
                        
                        ρρ = np.concatenate(([ρ[0]], ρ))
                        # center mass 
                        β = np.where(U > 0.0, 0.5, -0.5) 
                        ff[:-1, 1] +=  c[:-1] * dtP[:-1] * 1e5 * ΔV \
                            + ((β[1:  ] - 0.5) * ρ[1:  ] + (β[1:  ] + 0.5) * ρ[ :-1]) * U[1:  ] * A \
                            - ((β[ :-1] - 0.5) * ρ[ :-1] + (β[ :-1] + 0.5) * ρρ[ :-2]) * U[ :-1] * A
                        
                        # Staggered
                        Uc = 0.5 * (U[1:] + U[:-1])
                        dtPc = 0.5 * (dtP[:-2] + dtP[1:-1])
                        
                        β = np.where(Uc > 0.0, 0.5, -0.5)                     

                        # center momentum
                        ff[1:-1, 0] += U[1:-1] * c[1:-1] * dtPc * 1e5 * ΔV + ρf[1:-1] * dtU[1:-1] * ΔV \
                                     + ρ[ :-2] * Uc[1:  ] * A * ((β[1:  ] - 0.5) * U[2:  ] + (β[1:  ] + 0.5) * U[1:-1]) \
                                     - ρ[1:-1] * Uc[ :-1] * A * ((β[ :-1] - 0.5) * U[1:-1] + (β[ :-1] + 0.5) * U[ :-2]) \
                                     + (P[1:-1] - P[:-2]) * 1e5 * A + τw[1:-1] * (Sw / A) * ΔV
                        
                        # Momentum balance for half control volume
                        ff[-1, 0] += c[-1] * dtP[-1] * 1e5  * ΔV + ρf[-1] * dtU[-1] * ΔV * 0.5 \
                                   + ρ[-1] * U[-1] * A * U[-1] \
                                   - ρ[-1] * Uc[-1] * A * ((β[-1] - 0.5) * U[-1] + (β[-1] + 0.5) * U[-2]) \
                                   + (Ppresc - P[-2]) * 1e5 * A + τw[-1] * (Sw / A) * ΔV * 0.5
                       
                        # boundaries                        
                        ff[-1,1] = Ppresc - 0.5 * (P[-1] + P[-2])
                        ff[0,0] = Upresc - U[0] 

                        F.setArray(ff.flatten())                       
   


def transient_heat_transfer_1D(
    npipes, nx, dof,
    pipe_length,
    initial_time,
    final_time,
    initial_time_step,
    initial_solution,    
    impl_python=False
    ):
    
    # Time Stepper (TS) for ODE and DAE
    # DAE - https://en.wikipedia.org/wiki/Differential_algebraic_equation
    # https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/TS/
    ts = PETSc.TS().create()

    # http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/DM/index.html
    pipes = []
    for i in range(npipes):
        boundary_type = PETSc.DMDA.BoundaryType.GHOSTED
        da = PETSc.DMDA().create([nx], dof=dof, stencil_width=1, stencil_type='star', boundary_type=boundary_type)
        pipes.append(da)
    
    # Create a redundant DM, there is no petsc4py interface (yet)
    # so we created our own wrapper
    # http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/DM/DMREDUNDANT.html
#     dmredundant = PETSc.DM().create()
#     dmredundant.setType(dmredundant.Type.REDUNDANT)
#     CompositeSimple1D.redundantSetSize(dmredundant, 0, dof)
#     dmredundant.setDimension(1)
#     dmredundant.setUp()

    # http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/DM/DMCOMPOSITE.html
    dm = PETSc.DMComposite().create()
    
    for pipe in pipes:        
        dm.addDM(pipe)

#     dm.addDM(dmredundant)
#     CompositeSimple1D.compositeSetCoupling(dm)
    
    ts.setDM(dm)

    F = dm.createGlobalVec()

    if impl_python:        
        ode = Heat(dm, nx, dof, pipe_length)
        ts.setIFunction(ode.evalFunction, F)
    else:
        # http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/TS/TSSetIFunction.html
        assert False, 'C function not implemented yet!'
#         ts.setIFunction(CompositeSimple1D.formFunction, F,
#                          args=(conductivity, source_term, wall_length, temperature_presc))    
    
    x = dm.createGlobalVec()    

    x[...] = initial_solution

    # http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/TS/TSSetDuration.html
    ts.setDuration(max_time=final_time, max_steps=None)
    
    # http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/TS/TSSetInitialTimeStep.html
    ts.setInitialTimeStep(initial_time=initial_time, initial_time_step=initial_time_step)
    
    # http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/TS/TSSetProblemType.html
    ts.setProblemType(ts.ProblemType.NONLINEAR)
    
    
    ts.setFromOptions()

    ts.solve(x)

    return x

options = PETSc.Options()
options.clear()

dt = 1e-1             # [s]
dt_min = 1e-2           # [s]
dt_max = 10.0               # [s]

# time_intervals = [1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 0.2, 10]
time_intervals = [1.0]
# time_intervals = np.concatenate((np.logspace(-4, -0.1, num=200), [1.0])) 


# TS config
options.setValue('-ts_type', 'bdf') # https://en.wikipedia.org/wiki/Backward_differentiation_formula
options.setValue('-ts_bdf_order', 1) # https://en.wikipedia.org/wiki/Backward_differentiation_formula
options.setValue('-ts_adapt_type', 'basic') # basic or none
options.setValue('-ts_bdf_adapt', '')
options.setValue('-ts_adapt_dt_min', dt_min)
options.setValue('-ts_adapt_dt_max', dt_max)
options.setValue('-ts_exact_final_time', 'matchstep') # http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/TS/TSSetExactFinalTime.html
options.setValue('-ts_monitor', None)

dense = False
if dense:
    options.setValue('-dmcomposite_dense_jacobian', None)
else:
    options.delValue('-dmcomposite_dense_jacobian')

options.setValue('-ts_fd_color', None)

options.setValue('-mat_fd_type', 'ds')
# options.setValue('-mat_fd_coloring_err', 1e-5)
# options.setValue('-mat_fd_coloring_umin', 1e-12)

# options.setValue('-snes_linesearch_type', 'basic')
# options.setValue('-ts_max_snes_failures', -1)

# options.setValue('-mat_view', 'draw')
# options.setValue('-draw_pause', 500)
#options.setValue('-is_coloring_view', '')
options.setValue('-help', None)

options.setValue('-snes_monitor_short', None)
options.setValue('-snes_converged_reason', None)

options.setValue('-snes_type', 'newtonls')
# options.setValue('-npc_snes_type', 'ksponly')


# options.setValue('-snes_type', 'composite')
# options.setValue('-snes_composite_type', 'additiveoptimal')
# options.setValue('-snes_composite_sneses', 'nrichardson,newtonls')

# options.setValue('-snes_composite_sneses', 'fas,newtonls')
# options.setValue('-snes_composite_sneses', 'ngmres,newtonls')
 

# options.setValue('sub_0_snes_fas_levels', 4)
# options.setValue('sub_0_fas_levels_snes_type', 'gs')
# options.setValue('sub_0_fas_levels_snes_max_it', 6)
# options.setValue('sub_0_fas_coarse_snes_type', 'ngs')
# options.setValue('sub_0_fas_coarse_snes_max_it', 6)
# options.setValue('sub_0_fas_coarse_snes_linesearch_type', 'basic')
# options.setValue('sub_1_snes_linesearch_type', 'basic')
# options.setValue('sub_1_pc_type', 'mg')

# # for field split solver
# # for snes in ['sub_0_', 'sub_1_']: # for snes == composite 
# for snes in ['sub_0_']: # for snes == composite 
# # for snes in ['', 'npc_']:
# # for snes in ['npc_']:
#     # ksp config
#     options.setValue(snes + 'ksp_type', 'fgmres')  
#      
#     # pc config
#     # For direct solver
#     # options.setValue(snes + 'pc_type', 'lu')
#     # options.setValue(snes + 'ksp_type', 'preonly')
#     # options.setValue(snes + 'pc_factor_shift_type', 'NONZERO')
#     # options.setValue(snes + 'pc_factor_shift_amount', 1e-12)
#          
#     options.setValue(snes + 'pc_type', 'fieldsplit')  
#     options.setValue(snes + 'pc_fieldsplit_type', 'schur')  
#     options.setValue(snes + 'pc_fieldsplit_schur_fact_type', 'lower')   
#     options.setValue(snes + 'pc_fieldsplit_block_size', 2)   
#     options.setValue(snes + 'pc_fieldsplit_0_fields', 0)   
#     options.setValue(snes + 'pc_fieldsplit_1_fields', 1)   
#          
#     options.setValue(snes + 'fieldsplit_0_ksp_type', 'gmres')
#     options.setValue(snes + 'fieldsplit_0_pc_type', 'bjacobi')
# #     options.setValue(snes + 'fieldsplit_0_sub_pc_type', 'bjacobi')
#          
#          
#     options.setValue(snes + 'fieldsplit_1_pc_type', 'jacobi') 
#     options.setValue(snes + 'fieldsplit_1_pc_jacobi_type', 'diagonal') # (choose one of) DIAGONAL ROWMAX ROWSUM (PCJacobiSetType)
#     options.setValue(snes + 'fieldsplit_1_ksp_type', 'preonly') 
#          
#     # options.setValue(snes + 'fieldsplit_1_inner_pc_type', 'jacobi')
#     # options.setValue(snes + 'fieldsplit_1_inner_ksp_type', 'preonly')
#          
#     options.setValue(snes + 'fieldsplit_1_upper_ksp_type', 'preonly')
#     options.setValue(snes + 'fieldsplit_1_upper_pc_type', 'jacobi')
#     options.setValue(snes + 'fieldsplit_1_upper_pc_jacobi_type', 'diagonal') # (choose one of) DIAGONAL ROWMAX ROWSUM (PCJacobiSetType)
#          
#     options.setValue(snes + 'fieldsplit_1_mat_schur_complement_ainv_type', 'lump')   
#     # options.setValue(snes + 'pc_fieldsplit_schur_precondition', 'a11')   
#     options.setValue(snes + 'pc_fieldsplit_schur_precondition', 'selfp')   
#     # options.setValue(snes + 'pc_fieldsplit_detect_saddle_point', None)
#     # options.setValue(snes + 'pc_fieldsplit_default', None)
     

npipes = 1
nx = 1000
dof = 2
pipe_length = 100.0 # [m]

f, axarr = plt.subplots(2, sharex=True)
axarr[0].set_title('Velocity and Pressure')

initial_solution = np.zeros((nx,dof))
initial_solution[:,0] = 0.0001 # Velocity
initial_solution[:,1] = 1.0  # Pressure
initial_time = 0.0

sols = []
for final_time in time_intervals:

    sol = transient_heat_transfer_1D(
        npipes, nx, dof,
        pipe_length,
        initial_time,
        final_time,
        dt,
        initial_solution.flatten(),
        impl_python=True
        )
    
    SOL = sol[...].reshape(nx, dof)
    U = SOL[:,0]
    P = SOL[:,1]
    
    initial_solution[:,0] = U
    initial_solution[:,1] = P
    initial_time = final_time
    
    sols.append((U, P))
    
    dx = pipe_length / (nx - 1)
    x = np.linspace(0, npipes*pipe_length, npipes*nx, endpoint=True) + 0.5*dx
    xx = np.concatenate((x[:-1], [pipe_length]))


    UU = U
    PP = np.concatenate((P[:-1], [1.0]))
    axarr[0].cla()
    axarr[1].cla()
    axarr[0].plot(xx, UU, 'k.-')
    axarr[1].plot(xx, PP, 'r.-')
    plt.xlim(0, pipe_length)
    axarr[0].set_ylim(0, 40)
    axarr[1].set_ylim(0.9, 2)
    plt.show()
#     plt.draw()
#     plt.pause(0.0001)

