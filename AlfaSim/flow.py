import numpy as np
from petsc4py import PETSc
from physics0 import calculate_residualαUP
from physics1 import calculate_residualαUSP
from physics3 import calculate_residualαUSPsimple
from physics4 import calculate_residualαUPsimple
from models import density_model, computeGeometricProperties

calculate_residual = calculate_residualαUP


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
        
        
        # Auxiliary to be changed -> Build matrix structure
#         boundary_type = PETSc.DMDA.BoundaryType.GHOSTED
        dmda = PETSc.DMDA().create([nx], dof=dof, stencil_width=1, stencil_type='star')
        self.snes.setDM(dmda)
        
        self.F = dmda.createGlobalVec()
        self.X = dmda.createGlobalVec()
        self.Xold = dmda.createGlobalVec()
            
        self.snes.setFunction(self.form_function, self.F)
        self.snes.setUseFD() # Enables coloring, same as -snes_fd_color
        self.snes.setFromOptions()
        
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
        self.X[...] = initial_solution
        self.Xold[...] = initial_solution.copy()
        
    def set_duration(self, initial_time, final_time):
        self.initial_time = initial_time
        self.current_time = initial_time
        self.final_time = final_time
        
    def solve(self):        
        while self.current_time < self.final_time:
            print('************  \n  \tΔt = %gs\t t = %gs' % (self.Δt, self.current_time))
            
            max_iter = 10
            for i in range(max_iter):
                self.snes.solve(None, self.X)
                
                self.normalize_vol_frac()

                if self.snes.converged:
                    break
                else:
                    self.X = self.Xold.copy()
                    self.Δt = 0.5*self.Δt
                    
            self.Xold = self.X.copy()
            self.current_time += self.Δt
            
            self.update_ref_density()            
            
            if self.current_time > 0.001:
                self.Δt = np.maximum(self.min_Δt, np.minimum(self.Δt*1.1, self.max_Δt))
                    
                    
    def form_function(self, snes, X, F):       
        L   = self.L
        dof = self.dof
        nx  = self.nx
        dt = self.Δt
        dx = L / (nx - 1)

        U, α, P, dtU, dtα, dtP = self.get_UαP_array(X=X)

        residual = calculate_residual(dt, U, dtU, α, dtα, P, dtP, dx, nx, dof, self.Mpresc, self.Ppresc, 
                                      self.ρref, self.D, self.Dh, self.Sw, self.Si, None, None)
        F.setArray(residual.flatten())
               
    
    def get_UαP_array(self, X=None):
        nx  = self.nx
        nphases = self.nphases
        dof = self.dof
        dt = self.Δt
        
        uold = self.Xold.getArray()
        if X is None:
            u = self.X.getArray()     
        else:
            u = X.getArray(readonly=True)  
               
        u    =    u.reshape(nx, dof)        
        uold = uold.reshape(nx, dof)      
        udot = (u - uold) / dt
        
        U = u[:, 0:nphases]
        α = u[:, nphases:-1]
        P = u[:, -1]
      
        dtU = udot[:, 0:nphases]
        dtα = udot[:, nphases:-1]
        dtP = udot[:, -1]
        
        return U, α, P, dtU, dtα, dtP
                      
    def normalize_vol_frac(self):
        U, α, P, dtU, dtα, dtP = self.get_UαP_array()
        
        αG = α[:, 0].copy()
        αL = α[:, 1].copy()
        αTotal = αG + αL

        αfTotal = 0.5 * (αTotal[:-1] + αTotal[1:])               
        αfTotal = np.concatenate(([αfTotal[0]], αfTotal))    
            
        α[:, 0] = αG / αTotal
        α[:, 1] = αL / αTotal
        U[:, 0] *= αfTotal
        U[:, 1] *= αfTotal
        
    
    def update_ref_density(self):
        nphases = self.nphases
        dof = self.dof
        nx  = self.nx
        uold = self.Xold[...].reshape(nx, dof)  
        Pold = uold[:, -1]
        for phase in range(nphases):            
            self.ρref[:, phase] = density_model[phase](Pold*1e5)     
                   
                   
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
    
    return x, final_dt