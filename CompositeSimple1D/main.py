import sys
import numpy as np
import petsc4py
from petsc4py import PETSc
from matplotlib import pyplot as plt
import CompositeSimple1D



petsc4py.init(sys.argv)


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
                    dx_avg = 0.0
                    size = len(Fs)
                    Fs[size-1][0] = 0.0
                    
                    for X, X_t, F in zip(Xs[:-1], X_ts[:-1], Fs[:-1]):   

                        udot = X_t.getArray(readonly=True)
                        u    = X.getArray(readonly=True)
                        
                        udot = udot.reshape(dof, nx)
                        ∂tP, ∂tU = udot[0], udot[1]
                        
                        u = u.reshape(dof, nx)
                        P, U = u[0], u[1]
                        
                        D = 0.1 # [m]                        
                        ρ = 1.0
                        c = 0.0
                        Ppresc = 1.0e5 # [Pa]
                        Upresc = 0.1 # [m/s]
                        
                        A = 0.25 * np.pi * D ** 2 # [m]
                        dx = L / F.size
                        ΔV = A * dx
                        fw = 0.001
                        τw = 0.5 * fw * ρ * np.abs(U) * U  
                        Sw = np.pi * D
                        ff = np.zeros_like(u)
                        
                        # center mass
                        ff[0][:-1] += c * ∂tP * ΔV + ρ * U[1:] * A - ρ * U[:-1] * A
                        
                        # Staggered
                        Uc = 0.5 * (U[1:] + U[:-1])
                        
                        β = np.where(Uc > 0.0, 0.5, -0.5)                     
                        
                        # center momentum
                        ff[1][1:-1] += c * ∂tP * ΔV + ρ * ∂tU * ΔV \
                                     + ρ * Uc[1:  ] * A * ((β - 0.5) * U[2:-1] + (β + 0.5) * U[1:-2]) \
                                     - ρ * Uc[ :-1] * A * ((β - 0.5) * U[1:-2] + (β + 0.5) * U[ :-3]) \
                                     + (P[1:] - P[:-1]) * A + τw * Sw / A
                       
                        # boundaries                        
                        ff[0][-1] = Ppresc - 0.5 * (P[-1] + P[-2])
                        ff[1][0] = Upresc - U[0] 
                        ff[1][-1] = U[-2] - U[-1] 
                        
                        ff[ 0] = udot[       0]*dx - Q*dx - k * (u[       1] - u[       0])/dx
                        ff[-1] = udot[u.size-1]*dx - Q*dx + k * (u[u.size-1] - u[u.size-2])/dx
                        
                        # Internal Node
                        flux_term = k * (Xs[size-1][0] - u[u.size-1]) / dx
                        Fs[size-1][0] += +flux_term
                        ff[-1]  += -flux_term
                        
                        # BC's
                        ff[0] += +k * (u[0] - T) / (0.5*dx)
                        
                        F.setArray(ff.flatten())
                        dx_avg += dx
                    dx_avg /= len(Xs[:-1])
                    
                    Fs[size-1][0] += udot[size-1]*dx_avg - Q*dx_avg
   


def transient_heat_transfer_1D(
    npipes, nx, 
    pipe_length,
    final_time,
    initial_time_step,
    impl_python=False
    ):
    
    # Time Stepper (TS) for ODE and DAE
    # DAE - https://en.wikipedia.org/wiki/Differential_algebraic_equation
    # https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/TS/
    ts = PETSc.TS().create()

    dof = 2
    # http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/DM/index.html
    pipes = []
    for i in range(npipes):
        boundary_type = PETSc.DMDA.BoundaryType.GHOSTED
        da = PETSc.DMDA().create([nx], dof=dof, stencil_width=1, stencil_type='star', boundary_type=boundary_type)
        pipes.append(da)
    
    # Create a redundant DM, there is no petsc4py interface (yet)
    # so we created our own wrapper
    # http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/DM/DMREDUNDANT.html
    dmredundant = PETSc.DM().create()
    dmredundant.setType(dmredundant.Type.REDUNDANT)
    CompositeSimple1D.redundantSetSize(dmredundant, 0, dof)
    dmredundant.setDimension(1)
    dmredundant.setUp()

    # http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/DM/DMCOMPOSITE.html
    dm = PETSc.DMComposite().create()
    
    for pipe in pipes:        
        dm.addDM(pipe)

    dm.addDM(dmredundant)
    CompositeSimple1D.compositeSetCoupling(dm)
    
    ts.setDM(dm)

    F = dm.createGlobalVec()

    if impl_python:        
        ode = Heat(dm, pipe_length)
        ts.setIFunction(ode.evalFunction, F)
    else:
        # http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/TS/TSSetIFunction.html
        assert False, 'C function not implemented yet!'
#         ts.setIFunction(CompositeSimple1D.formFunction, F,
#                          args=(conductivity, source_term, wall_length, temperature_presc))    
    
    x = dm.createGlobalVec()
    
#     x[...] = initial_temperature

    # http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/TS/TSSetDuration.html
    ts.setDuration(max_time=final_time, max_steps=None)
    
    # http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/TS/TSSetExactFinalTime.html
    ts.setExactFinalTime(ts.ExactFinalTimeOption.STEPOVER)
    
    # http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/TS/TSSetInitialTimeStep.html
    ts.setInitialTimeStep(initial_time=0.0, initial_time_step=initial_time_step)
    
    # http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/TS/TSSetProblemType.html
    ts.setProblemType(ts.ProblemType.NONLINEAR)

    ts.setFromOptions()

    ts.solve(x)

    return x

options = PETSc.Options()
options.clear()

dt = 0.001                 # [s]
dt_min = 1e-4              # [s]
dt_max = 0.1               # [s]

#ts_type = "beuler"
#ts_type = "pseudo" # http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/TS/TSPSEUDO.html
#ts_type = "cn"
ts_type = "bdf"
options.setValue('-ts_type', ts_type)

options.setValue('-ts_bdf_order', 3) # https://en.wikipedia.org/wiki/Backward_differentiation_formula
options.setValue('-ts_adapt_type', 'basic') # basic or none
options.setValue('-ts_bdf_adapt', '')
options.setValue('-ts_adapt_dt_min', dt_min)
options.setValue('-ts_adapt_dt_max', dt_max)
options.setValue('-ts_monitor', None)

dense = False
if dense:
    options.setValue('-dmcomposite_dense_jacobian', None)
else:
    options.delValue('-dmcomposite_dense_jacobian')

options.setValue('-ts_fd_color', None)

# options.setValue('-mat_view', 'draw')
# options.setValue('-draw_pause', 300)
#options.setValue('-is_coloring_view', '')
# options.setValue('-help', None)

npipes = 2
nx = 5
pipe_length = 1.0          # [m]

#time_intervals = [0.001, 0.01, 0.05, 0.1, 1.0]
time_intervals = [1.0]
sols = []
for final_time in time_intervals:
    sol = transient_heat_transfer_1D(
        npipes, nx, 
        pipe_length,
        final_time,
        dt,
        impl_python=True
        )
    sols.append(sol[...])
    
x = np.linspace(0, npipes*pipe_length, npipes*nx + 1)
for sol in sols:
#     plt.plot(x, sol)
    plt.plot(x, np.concatenate((sol[0:nx], [sol[npipes*nx]], sol[nx:-1][::-1])))
plt.show()