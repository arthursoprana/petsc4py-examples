import numpy as np
from petsc4py import PETSc
from physics0 import calculate_residualαUP
from physics1 import calculate_residualαUSP
from physics3 import calculate_residualαUSPsimple
from physics4 import calculate_residualαUPsimple
from models import density_model, computeGeometricProperties
from physics5 import calculate_residual_mass, calculate_residual_mom

calculate_residual = calculate_residualαUPsimple


class Solver(object):
    def __init__(self, nx, dof, pipe_length, diameter, nphases, α0, mass_flux_presc, pressure_presc):       
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
        
        self.snes = PETSc.SNES().create()
        self.snes_mom = PETSc.SNES().create()
        
        
        # Auxiliary to be changed -> Build matrix structure
#         boundary_type = PETSc.DMDA.BoundaryType.GHOSTED
        dmda_mass = PETSc.DMDA().create([nx], dof=dof-nphases, stencil_width=1, stencil_type='star')
        dmda_mom  = PETSc.DMDA().create([nx], dof=nphases, stencil_width=1, stencil_type='star')
        
        self.snes.setDM(dmda_mass)
        self.snes_mom.setDM(dmda_mom)
        
        self.F = dmda_mass.createGlobalVec()
        self.X = dmda_mass.createGlobalVec()
        self.Xold = dmda_mass.createGlobalVec()
  
        self.Fmom = dmda_mom.createGlobalVec()
        self.Xmom = dmda_mom.createGlobalVec()
        self.Xmomold = dmda_mom.createGlobalVec()
                  
        self.snes.setUpdate(self.update_function)
        self.snes.setFunction(self.form_function, self.F)
        self.snes.setUseFD() # Enables coloring, same as -snes_fd_color
        self.snes.setFromOptions()
        
        self.snes_mom.setFunction(self.form_function_mom, self.Fmom)
        self.snes_mom.setUseFD() # Enables coloring, same as -snes_fd_color
        self.snes_mom.setFromOptions()
        
        self.initial_time = 0.0
        self.current_time = 0.0
        self.final_time = 0.0
        self.Δt  = 0.0
        
    def set_initial_timestep(self, initial_timestep):
        self.Δt = initial_timestep
        
    def set_min_timestep(self, min_timestep):
        self.min_Δt = min_timestep
        
    def set_max_timestep(self, max_timestep):
        self.max_Δt = max_timestep
  
    def set_initial_solution(self, initial_solution):
        u = initial_solution.reshape(self.nx, self.dof)        
        U = u[:, 0:self.nphases]
        α = u[:, self.nphases:-1]
        P = u[:, -1]
        
        initial_solution_mass = np.hstack((α, P[:, np.newaxis]))
        self.X[...] = initial_solution_mass.flatten()
        self.Xold[...] = self.X[...].copy()
        
        self.Xmom[...] = U.flatten()
        self.Xmomold[...] = self.Xmom[...].copy()
        
        self.U = U
        self.Uold = U.copy()
        
    def set_duration(self, initial_time, final_time):
        self.initial_time = initial_time
        self.current_time = initial_time
        self.final_time = final_time
        
    def solve(self): 
        nphases = self.nphases
        
        dof = self.dof
        nx  = self.nx   
        
        
#         self.snes_mom.setType(self.snes_mom.Type.KSPONLY)
#         self.snes.setType(self.snes.Type.KSPONLY)
        # CHUNCHO!
#         options = PETSc.Options()
#         options.setValue('-snes_type', 'composite')
#         options.setValue('-snes_composite_type', 'additiveoptimal')
#         options.setValue('-snes_composite_sneses', 'newtonls')
#         options.setValue('-snes_composite_damping', (0.3))
#         options.setValue('-sub_0_snes_linesearch_type', 'basic')
#         self.snes.setFromOptions()
#         
        while self.current_time < self.final_time:
            print('************  \n  \tΔt = %gs\t t = %gs' % (self.Δt, self.current_time))
            max_iter = 10
            for i in range(max_iter):
                
                for j in range(5):
                    print('Solve mom')
                    self.snes_mom.solve(None, self.Xmom)                    
                    print('Solve mass')
                    self.snes.solve(None, self.X)
                    
                    sol = self.X[...]
                    u = sol.reshape(nx, dof-nphases)
                    α = u[:, :-1] # view!
                    α[:, 0] = np.maximum(α[:, 0], 0.0)
                    α[:, 1] = np.minimum(α[:, 1], 1.0)    
                    αG = α[:, 0]                 
                    αL = α[:, 1]                 
                    α[:, 0] = αG / (αG + αL)
                    α[:, 1] = αL / (αG + αL)
                
                if self.snes.converged:
                    break
                else:
                    self.X = self.Xold.copy()
                    self.Δt = 0.5*self.Δt
                    
            self.Xold = self.X.copy()
            self.Xmomold = self.Xmom.copy()
            self.current_time += self.Δt
            
            if self.current_time > 0.001:
                self.Δt = np.maximum(self.min_Δt, np.minimum(self.Δt*1.1, self.max_Δt))
            
    def update_function(self, snes, step):
        
        nphases = self.nphases
        
        dof = self.dof
        nx  = self.nx   
   
        sol = snes.getSolution()[...]
        u = sol.reshape(nx, dof-nphases)         
        #U = u[:, 0:nphases]
        α = u[:, :-1] # view!
        P = u[:, -1]
        
#         α[:, 0] = np.maximum(α[:, 0], 0.0)
#         α[:, 1] = np.minimum(α[:, 1], 1.0)
#         
#         α[:, 0] = α[:, 0] / (α[:, 0] + α[:, 1])
#         α[:, 1] = α[:, 1] / (α[:, 0] + α[:, 1])
        
#         (Mpresc[phase] - 0.001 * ρf[0] * U[0] * A)
        
        # Compute ref density
        uold = self.Xold[...].reshape(nx, dof-nphases)  
        Pold = uold[:, -1]
        for phase in range(nphases):            
            self.ρref[:, phase] = density_model[phase](Pold*1e5)
            
        # Compute geometric properties
        self.Dh, self.Sw, self.Si, self.H = computeGeometricProperties(α, self.D)
        
        # Correct velocities


        
        
    def form_function(self, snes, X, F):       
        L   = self.L
        dof = self.dof
        nx  = self.nx
        nphases = self.nphases
        dt = self.Δt

        uold = self.Xold.getArray()
        u    = X.getArray(readonly=True)        
        uold = uold.reshape(nx, dof-nphases)        
        αold = uold[:, :-1]
        Pold = uold[:,  -1]
        
        u = u.reshape(nx, dof-nphases)
        
        α = u[:, :-1]
        P = u[:,  -1]

        # getting from mom
        uold = self.Xmomold.getArray()
        u    = self.Xmom.getArray(readonly=True)       
        uold = uold.reshape(nx, nphases)        
        Uold = uold.copy()        
        u = u.reshape(nx, nphases)        
        U = u.copy()
        
        dx = L / (nx - 1)

        residual = calculate_residual_mass(dt, U, Uold, α, αold, P, Pold, dx, nx, dof, self.Mpresc, self.Ppresc, 
                                      self.ρref, self.D, self.Dh, self.Sw, self.Si, None, None)
        F.setArray(residual.flatten())
                      
    def form_function_mom(self, snes, X, F):       
        L   = self.L
        dof = self.dof
        nx  = self.nx
        nphases = self.nphases
        dt = self.Δt
        

        uold = self.Xmomold.getArray()
        u    = X.getArray(readonly=True)
     
        uold = uold.reshape(nx, nphases)        
        Uold = uold.copy()        
        u = u.reshape(nx, nphases)        
        U = u.copy()

        dx = L / (nx - 1)        
        
        # getting from mass
        uold = self.Xold.getArray()
        u    = self.X.getArray(readonly=True)      
        uold = uold.reshape(nx, dof-nphases)        
        αold = uold[:, :-1]
        Pold = uold[:,  -1]        
        u = u.reshape(nx, dof-nphases)        
        α = u[:, :-1]
        P = u[:,  -1]
        
        residual = calculate_residual_mom(dt, U, Uold, α, αold, P, Pold, dx, nx, dof, self.Mpresc, self.Ppresc, 
                                      self.ρref, self.D, self.Dh, self.Sw, self.Si, None, None)
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
        
    α0 = initial_solution.reshape((nx,dof))[:, nphases:-1]
    solver = Solver(nx, dof, pipe_length, diameter, nphases, α0, mass_flux_presc, pressure_presc)
    solver.set_initial_solution(initial_solution)  

    solver.set_duration(initial_time, final_time)

    solver.set_initial_timestep(initial_time_step)
    solver.set_min_timestep(dt_min)
    solver.set_max_timestep(dt_max)    
    
    solver.solve()  
    
    final_dt = solver.Δt
    
    x = solver.X
    xmom = solver.Xmom
    return x, xmom, final_dt