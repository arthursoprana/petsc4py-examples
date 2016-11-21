import sys
import numpy as np
import petsc4py
from petsc4py import PETSc
from matplotlib import pyplot as plt
import HeatTransfer1D

petsc4py.init(sys.argv)

def transient_heat_transfer_1D(
    nx, temperature_left, 
    temperature_right, 
    conductivity,
    source_term,
    wall_length,
    final_time,
    initial_time_step
    ):
    
    # Time Stepper (TS) for ODE and DAE
    # DAE - https://en.wikipedia.org/wiki/Differential_algebraic_equation
    # https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/TS/
    ts = PETSc.TS().create()

    # http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/DM/index.html
    #
    da1 = PETSc.DMDA().create([nx],dof=1, stencil_width=1, stencil_type='star')
    da2 = PETSc.DMDA().create([nx],dof=1, stencil_width=1, stencil_type='star')
    
	# Create a redundant DM, there is no petsc4py interface (yet)
	# so we created our own wrapper
    dmredundant = PETSc.DM().create()
    dmredundant.setType(dmredundant.Type.REDUNDANT)
    HeatTransfer1D.redundantSetSize(dmredundant, 0, 1)
    dmredundant.setDimension(1)
    dmredundant.setUp()

    dm = PETSc.DMComposite().create()
    dm.addDM(da1)
    dm.addDM(da2)
    dm.addDM(dmredundant)
    HeatTransfer1D.compositeSetCoupling(dm)
    
    ts.setDM(dm)

    F = dm.createGlobalVec()

    # http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/TS/TSSetIFunction.html
    ts.setIFunction(HeatTransfer1D.formFunction, F,
                     args=(temperature_left, temperature_right, conductivity, source_term, wall_length))

    x = dm.createGlobalVec()
    
    HeatTransfer1D.formInitGuess(x, dm, temperature_left, temperature_right, conductivity, source_term, wall_length)

    # http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/TS/TSSetDuration.html
    ts.setDuration(max_time=final_time, max_steps=None)
    
    # http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/TS/TSSetExactFinalTime.html
    ts.setExactFinalTime(ts.ExactFinalTimeOption.STEPOVER)
    
    # http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/TS/TSSetInitialTimeStep.html
    ts.setInitialTimeStep(initial_time=0.0, initial_time_step=initial_time_step)
    
    # http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/TS/TSSetProblemType.html
    ts.setProblemType(ts.ProblemType.NONLINEAR)
    
    # Another way to set the solve type is through PETSc.Options()
    #ts.setType(ts.Type.CRANK_NICOLSON)
    #ts.setType(ts.Type.THETA)
    #ts.setTheta(theta=0.9999)
    #ts.setType(ts.Type.EIMEX) # http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/TS/TSEIMEX.html
    #ts.setType(ts.Type.BDF      )

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
#options.delValue('-ts_bdf_adapt')

#options.setValue('-is_coloring_view', '')

nx = 50
temperature_left  = 0.0    # [degC]
temperature_right = 50.0   # [degC]
conductivity = 1.0         # [W/(m.K)]
source_term = 100.0          # [W/m3]
wall_length = 1.0          # [m]

time_intervals = [0.001, 0.01, 0.05, 0.1, 1.0]
#time_intervals = [0.001]
sols = []
for final_time in time_intervals:
    sol = transient_heat_transfer_1D(
        nx, temperature_left, 
        temperature_right, 
        conductivity,
        source_term,
        wall_length,
        final_time,
        dt
        )
    sols.append(sol[...])
    
x = np.linspace(0, wall_length, 2*nx + 1)
for sol in sols:
    plt.plot(x, sol)
plt.show()