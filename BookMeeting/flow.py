import numpy as np
from petsc4py import PETSc
from physics0 import calculate_residualαUP
from physics1 import calculate_residualαUSP
from models import density_model, computeGeometricProperties

calculate_residual = calculate_residualαUSP


class Flow(object):
    def __init__(self, dm, nx, dof, pipe_length, diameter, nphases, α0, mass_flux_presc, pressure_presc):
        self.dm  = dm        
        self.L   = pipe_length
        self.nx  = nx
        self.dof = dof
        self.nphases = nphases
        self.D =  diameter 
        self.Mpresc = mass_flux_presc
        self.Ppresc = pressure_presc
        self.ρref = np.zeros((nx, nphases))
        for phase in range(nphases):
            self.ρref[:, phase] = density_model[phase](1e5)

        self.Dh, self.Sw, self.Si, self.H = computeGeometricProperties(α0, self.D)
    
    def updateFunction(self, snes, step):
        
        nphases = self.nphases
        
        dof = self.dof
        nx  = self.nx   
   
        sol = snes.getSolution()[...]
        u = sol.reshape(nx, dof)         
        #U = u[:, 0:nphases]
        α = u[:, nphases:-1]
        α = np.maximum(α, 1e-5)
        α = np.minimum(α, 1.0)
        P = u[:, -1]
    
        # Compute ref density
        for phase in range(nphases):
            self.ρref[:, phase] = density_model[phase](P*1e5)
            
        # Compute geometric properties
        self.Dh, self.Sw, self.Si, self.H = computeGeometricProperties(α, self.D)


        
    def evalFunction(self, ts, t, x, xdot, f):
        dm  = self.dm
        L   = self.L
        dof = self.dof
        nx  = self.nx
        nphases = self.nphases
        dt = ts.getTimeStep()
        
        with dm.getAccess(x, locs=None) as Xs:
            with dm.getAccess(xdot, locs=None) as X_ts:
                with dm.getAccess(f, locs=None) as Fs:
                    
                    for X, X_t, F in zip(Xs[:], X_ts[:], Fs[:]):   
                        udot = X_t.getArray(readonly=True)
                        u    = X.getArray(readonly=True)
                        
                        udot = udot.reshape(nx, dof)
                        
                        dtU = udot[:, 0:nphases]
                        dtα = udot[:, nphases:-1]
                        dtP = udot[:, -1]
                        
                        u = u.reshape(nx, dof)
                        
                        U = u[:, 0:nphases]
                        α = u[:, nphases:-1]
                        P = u[:, -1]

                        dx = L / (nx - 1)
            
                        residual = calculate_residual(dt, U, dtU, α, dtα, P, dtP, dx, nx, dof, self.Mpresc, self.Ppresc, 
                                                      self.ρref, self.D, self.Dh, self.Sw, self.Si, self.H, None)
                        F.setArray(residual.flatten())
                        
def transient_pipe_flow_1D(
    npipes, nx, dof, nphases,
    pipe_length, diameter,
    initial_time,
    final_time,
    initial_time_step,
    dt_min,
    dt_max,
    initial_solution,    
    mass_flux_presc,
    pressure_presc,
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
    
    ts.setDM(dm)
        
    F = dm.createGlobalVec()

    if impl_python:     
        α0 = initial_solution.reshape((nx,dof))[:, nphases:-1]
        pde = Flow(dm, nx, dof, pipe_length, diameter, nphases, α0, mass_flux_presc, pressure_presc)
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
    
    prev_sol = initial_solution.copy()
    
#     restart = PreStep(prev_sol)
#     ts.setPreStep(restart.prestep)
#     ts.setPostStep(restart.poststep)
    
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
    
#     while ts.diverged:
#         x[...] = prev_sol.copy()
#         ts.setInitialTimeStep(initial_time=initial_time, initial_time_step=0.5*initial_time_step)
#         ts.solve(x)
    
    final_dt = ts.getTimeStep()
    
    return x, final_dt