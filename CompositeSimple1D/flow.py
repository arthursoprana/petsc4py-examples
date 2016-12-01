import numpy as np
import CompositeSimple1D
from petsc4py import PETSc
from physics0 import Flow0
from physics1 import Flow1

Flow = Flow0

class MyTS:
    def __init__(self):
        self.log = {}
    def _log(self, method, *args):
        self.log.setdefault(method, 0)
        self.log[method] += 1

    def create(self, ts, *args):
        self._log('create', *args)
        self.vec_update = PETSc.Vec()

    def destroy(self, ts, *args):
        self._log('destroy', *args)
        self.vec_update.destroy()

    def setFromOptions(self, ts, *args):
        self._log('setFromOptions', *args)

    def setUp(self, ts, *args):
        self._log('setUp', ts, *args)
        self.vec_update = ts.getSolution().duplicate()

    def reset(self, ts, *args):
        self._log('reset', ts, *args)

    def solveStep(self, ts, t, u, *args):
        self._log('solveStep', ts, t, u, *args)
        ts.snes.solve(None, u)

    def adaptStep(self, ts, t, u, *args):
        self._log('adaptStep', ts, t, u, *args)
        nx = 1000
        dof = 5
        nphases = 2
        uu = u[...].reshape(nx, dof)         
        U = uu[:, 0:nphases]
        cfl = 0.8
        Umax = np.max(np.abs(U))
        old_dt = ts.getTimeStep()
        L = 1000
        dx = L / (nx - 1)
        new_dt = cfl * dx / Umax
        dt_min = 1e-4
        dt_max = 10.0
        
        if ts.diverged:
            dt = np.maximum(dt_min, 0.5*old_dt)
        else:
            if t > 0.1:                
                dt = np.maximum(dt_min, np.minimum(np.minimum(new_dt, dt_max), 1.1*old_dt))
            else:
                dt = ts.getTimeStep()
        return (dt, True)
    



def transient_pipe_flow_1D(
    npipes, nx, dof, nphases,
    pipe_length,
    initial_time,
    final_time,
    initial_time_step,
    dt_min,
    dt_max,
    initial_solution,    
    impl_python=False
    ):
    
    # Time Stepper (TS) for ODE and DAE
    # DAE - https://en.wikipedia.org/wiki/Differential_algebraic_equation
    # https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/TS/
    ts = PETSc.TS().create()
    ts.createPython(MyTS(), comm=PETSc.COMM_SELF)
    
    # http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/DM/index.html
    pipes = []
    for i in range(npipes):
        boundary_type = PETSc.DMDA.BoundaryType.GHOSTED
        da = PETSc.DMDA().create([nx], dof=dof, stencil_width=1, stencil_type='star', boundary_type=boundary_type)
        da.setFromOptions()
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
    CompositeSimple1D.registerNewSNES()
    
    ts.setDM(dm)
        
    F = dm.createGlobalVec()

    if impl_python:     
        α0 = initial_solution.reshape((nx,dof))[:, nphases:-1]
        pde = Flow(dm, nx, dof, pipe_length, nphases, α0)
        ts.setIFunction(pde.evalFunction, F)
    else:
        # http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/TS/TSSetIFunction.html
        assert False, 'C function not implemented yet!'
#         ts.setIFunction(CompositeSimple1D.formFunction, F,
#                          args=(conductivity, source_term, wall_length, temperature_presc))    
    
    snes = ts.getSNES()
    
    snes.setUpdate(pde.updateFunction)
    
    x = dm.createGlobalVec()    

    x[...] = initial_solution

    # http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/TS/TSSetDuration.html
    ts.setDuration(max_time=final_time, max_steps=None)
    
    # http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/TS/TSSetInitialTimeStep.html
    ts.setInitialTimeStep(initial_time=initial_time, initial_time_step=initial_time_step)
    
    # http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/TS/TSSetProblemType.html
    ts.setProblemType(ts.ProblemType.NONLINEAR)
    ts.setEquationType(ts.EquationType.IMPLICIT)
    
#     ts.setPreStep(prestep, args=None, kargs=None)
    
    options = PETSc.Options()
    options.setValue('-ts_adapt_dt_min', dt_min)
    options.setValue('-ts_adapt_dt_max', dt_max)
    
    ts.setFromOptions()

    snes = ts.getSNES()
    if options.getString('snes_type') in [snes.Type.VINEWTONRSLS, snes.Type.VINEWTONSSLS]:
#     if True:
#         snesvi = snes.getCompositeSNES(1)
        snesvi = snes
        
        xl = np.zeros((nx, dof))
        xl[:,:nphases] = -100
        xl[:,-1] = 0
        xl[:, nphases:-1] = 0

        
        xu = np.zeros((nx, dof))    
        xu[:,:nphases] =  100
        xu[:,-1] = 1000
        xu[:, nphases:-1] = 1
             
        xlVec = dm.createGlobalVec()
        xuVec = dm.createGlobalVec()   
        xlVec.setArray(xl.flatten())
        xuVec.setArray(xu.flatten())    

        snesvi.setVariableBounds(xlVec, xuVec)
        
    ts.solve(x)
    
    final_dt = ts.getTimeStep()
    
    return x, final_dt