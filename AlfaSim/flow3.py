import numpy as np
from petsc4py import PETSc
from models import density_model, computeGeometricProperties
from physics6 import calculate_residual_mass, calculate_residual_mom,\
    calculate_velocity_update, calculate_coeff_mom, calculate_residual_press

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
        
        self.snes_press = PETSc.SNES().create()
        self.snes_mass  = PETSc.SNES().create()
        self.snes_mom   = PETSc.SNES().create()
        
        # Auxiliary to be changed -> Build matrix structure
#         boundary_type = PETSc.DMDA.BoundaryType.GHOSTED
        dmda_press = PETSc.DMDA().create([nx], dof=1, stencil_width=1, stencil_type='star')
        dmda_mass  = PETSc.DMDA().create([nx], dof=nphases, stencil_width=1, stencil_type='star')
        dmda_mom   = PETSc.DMDA().create([nx], dof=nphases, stencil_width=1, stencil_type='star')
        
        self.snes_press.setDM(dmda_press)
        self.snes_mass.setDM(dmda_mass)
        self.snes_mom.setDM(dmda_mom)
  
        self.Fpress = dmda_press.createGlobalVec()
        self.XΔpress = dmda_press.createGlobalVec()
        self.Xpress = dmda_press.createGlobalVec()
        self.Xpressold = dmda_press.createGlobalVec()
        
        self.Fmass = dmda_mass.createGlobalVec()
        self.Xmass = dmda_mass.createGlobalVec()
        self.Xmassold = dmda_mass.createGlobalVec()
        
        self.Fmom = dmda_mom.createGlobalVec()
        self.Xmom = dmda_mom.createGlobalVec()
        self.Xmomold = dmda_mom.createGlobalVec()

        self.snes_press.setFunction(self.form_function_press, self.Fpress)
        self.snes_press.setUseFD() # Enables coloring, same as -snes_fd_color
        self.snes_press.setFromOptions()
        
        self.snes_mass.setFunction(self.form_function_mass, self.Fmass)
        self.snes_mass.setUseFD() # Enables coloring, same as -snes_fd_color
        self.snes_mass.setFromOptions()
        
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

        self.XΔpress[...] = 0 # initial pressure correction is zero
        
        self.Xpress[...] = P
        self.Xpressold[...]  = self.Xpress[...].copy()
        
        self.Xmass[...] = α.flatten()
        self.Xmassold[...]  = self.Xmass[...].copy()
        
        self.Xmom[...] = U.flatten()
        self.Xmomold[...] = self.Xmom[...].copy()
        
        self.U = U
        self.Uold = U.copy()
        
        self.Ap_u = np.zeros_like(U)
        
    def set_duration(self, initial_time, final_time):
        self.initial_time = initial_time
        self.current_time = initial_time
        self.final_time = final_time
        
    def solve(self): 
        nphases = self.nphases
        
        dof = self.dof
        nx  = self.nx  
        
#         self.snes_mom.setType(self.snes_mom.Type.KSPONLY)
#         self.snes_mass.setType(self.snes_mass.Type.KSPONLY)
#         self.snes_press.setType(self.snes.Type.KSPONLY)

        self.snes_press.setTolerances(rtol=1e-50, stol=1e-50, atol=1e-8, max_it=10)
        self.snes_mass.setTolerances(rtol=1e-50, stol=1e-50, atol=1e-8, max_it=10)
        self.snes_mom.setTolerances(rtol=1e-50, stol=1e-50, atol=1e-6, max_it=10)
        
        tolerance = 1e-6
        max_iterations = 10
        while self.current_time < self.final_time:
            print('************  \n  \tΔt = %gs\t t = %gs' % (self.Δt, self.current_time))
            max_breaks = 10

            for i in range(max_breaks):
                print('****************** RESIDUAL ******************')
                print('Total Volume  |  Total Mass  |  Total Momentum')
                outer_iteration = 0
                residual_press = 1e20
                residual_mass  = 1e20
                residual_mom  = 1e20
                while (residual_mom > tolerance or residual_press > tolerance or residual_mass > tolerance) and outer_iteration < max_iterations:  
                    self.snes_mom.solve(None, self.Xmom)                     
                    self.calc_coeff_mom()
                    
                    residual_press = 1e20
                    residual_mass  = 1e20
                    inner_iteration = 0
                    
                    while (residual_press > tolerance or residual_mass > tolerance) and inner_iteration < max_iterations:  
                        self.XΔpress[...] = 0.0 # This seems necessary, why?
                              
                        self.snes_press.solve(None, self.XΔpress)
                        
                        ΔP = self.XΔpress[...]
                        self.correct_velocities(ΔP)
                        self.correct_pressure(ΔP, relax_factor=1.0) 
                        
                        self.snes_mass.solve(None, self.Xmass)  
                                      
                        self.normalize_vol_frac()
                        
                        self.form_function_press(self.snes_press, self.XΔpress, self.Fpress)                        
                        residual_mass  = self.snes_mass.getFunction()[0].norm()                        
                        residual_press = self.snes_press.getFunction()[0].norm()
                        
                        inner_iteration += 1
                    
                    residual_mom = self.calculate_residual_mom()
                    
                    print('%.4e \t %.4e \t %.4e \t %i  %i' %
                          (residual_mass, residual_press, residual_mom, outer_iteration, inner_iteration))
                    
                    outer_iteration += 1
                if self.snes_mass.converged and self.snes_press.converged:
                    break
                else:
                    print('\t\t ******* BREAKING TIMESTEP %i *******' % i)
                    self.Xpress = self.Xpressold.copy()
                    self.Xmass = self.Xmassold.copy()
                    self.Xmom = self.Xmomold.copy()
                    self.XΔpress[...] = 0.0
                    self.Δt = 0.5*self.Δt
                    
            self.Xpressold = self.Xpress.copy()
            self.Xmassold  = self.Xmass.copy()
            self.Xmomold   = self.Xmom.copy()
            
            self.current_time += self.Δt
            
            # Update ρref
            Pold = self.Xpressold[...]
            for phase in range(nphases):            
                self.ρref[:, phase] = density_model[phase](Pold*1e5)
            
            if self.current_time > 0.001:
                self.Δt = np.maximum(self.min_Δt, np.minimum(self.Δt*1.1, self.max_Δt))
    
    def get_velocity_array(self):
        nx  = self.nx
        nphases = self.nphases
        uold = self.Xmomold.getArray()
        u    = self.Xmom.getArray()       
        uold = uold.reshape(nx, nphases)        
        Uold = uold.copy()              
        U = u.reshape(nx, nphases) # view!
        return U, Uold
    
    def get_vol_frac_array(self):
        nx  = self.nx
        nphases = self.nphases
        uold = self.Xmassold.getArray()
        u    = self.Xmass.getArray()       
        uold = uold.reshape(nx, nphases)        
        αold = uold.copy()              
        α = u.reshape(nx, nphases) # view!
        return α, αold
    
    def get_pressure_array(self):
        Pold = self.Xpressold.getArray()
        P    = self.Xpress.getArray()       
        return P, Pold
    
    def form_function_press(self, snes, X, F):      
        
        F[...] = 0.0 # For safety

        L   = self.L
        dof = self.dof
        nx  = self.nx
        dt = self.Δt
        
        ΔP = X.getArray(readonly=True)
              
        P, Pold = self.get_pressure_array()       
        α, αold = self.get_vol_frac_array()        
        U, Uold = self.get_velocity_array()
        
        dx = L / (nx - 1)
                    
        residual = calculate_residual_press(dt, U, Uold, α, αold, P + ΔP, Pold, ΔP, dx, nx, dof, self.Mpresc, self.Ppresc, 
                                      self.ρref, self.D, self.Dh, self.Sw, self.Si, None, None, None, self.Ap_u)
        F.setArray(residual.flatten())
                
    def form_function_mass(self, snes, X, F):      
        
        F[...] = 0.0 # For safety

        L   = self.L
        dof = self.dof
        nx  = self.nx
        nphases = self.nphases
        dt = self.Δt

        uold = self.Xmassold.getArray()
        u    = X.getArray(readonly=True)        
        αold = uold.reshape(nx, nphases)
        α = u.reshape(nx, nphases)
        
        P, Pold = self.get_pressure_array()
        U, Uold = self.get_velocity_array()
        
        dx = L / (nx - 1)
            
        residual = calculate_residual_mass(dt, U, Uold, α, αold, P, Pold, dx, nx, dof, self.Mpresc, self.Ppresc, 
                                      self.ρref, self.D, self.Dh, self.Sw, self.Si, None, None, None, self.Ap_u)
        F.setArray(residual.flatten())

        
    def form_function_mom(self, snes, X, F):       
        F[...] = 0.0 # For safety
        
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
        
        P, Pold = self.get_pressure_array()
        α, αold = self.get_vol_frac_array()      
        
        residual = calculate_residual_mom(dt, U, Uold, α, αold, P, Pold, dx, nx, dof, self.Mpresc, self.Ppresc, 
                                      self.ρref, self.D, self.Dh, self.Sw, self.Si, None, None, self.Ap_u)
        F.setArray(residual.flatten())   
 
    def calc_coeff_mom(self):
        L   = self.L
        dof = self.dof
        nx  = self.nx
        nphases = self.nphases
        dt = self.Δt                 
    
        uold = self.Xmomold.getArray()
        u    = self.Xmom.getArray()                  
        uold = uold.reshape(nx, nphases)        
        Uold = uold.copy()        
        u = u.reshape(nx, nphases)        
        U = u.copy()
    
        dx = L / (nx - 1)        
        
        P, Pold = self.get_pressure_array()
        α, αold = self.get_vol_frac_array()  
        
        calculate_coeff_mom(dt, U, Uold, α, αold, P, Pold, dx, nx, dof, self.Mpresc, self.Ppresc, 
             self.ρref, self.D, self.Dh, self.Sw, self.Si, None, None, self.Ap_u) 

    def correct_velocities(self, ΔP):        
        L   = self.L
        dof = self.dof
        nx  = self.nx
        dt = self.Δt  
        dx = L / (nx - 1)       
    
        U, Uold = self.get_velocity_array()       
        P, Pold = self.get_pressure_array()
        α, αold = self.get_vol_frac_array()  
    
        ΔU = calculate_velocity_update(ΔP, None, dt, U, Uold, α, αold, P, Pold, dx, nx, dof, self.Mpresc, self.Ppresc, self.ρref, self.D, Ap_uT=self.Ap_u)
        U[:] += ΔU   
        
    def correct_pressure(self, ΔP, relax_factor=1.0):    
        P, _ = self.get_pressure_array()
        P[:] += relax_factor*ΔP  
    
    def normalize_vol_frac(self):
        α, _ = self.get_vol_frac_array() 
        
        αG = α[:, 0].copy()                
        αL = α[:, 1].copy()     
        αTotal = αG + αL
        α[:, 0] = αG / αTotal
        α[:, 1] = αL / αTotal
    
    def calculate_residual_mom(self):
        self.form_function_mom(self.snes_mom, self.Xmom, self.Fmom)  
        
        U, _ = self.get_velocity_array()       
        Ap_u = self.Ap_u
        
        denominator = np.linalg.norm(Ap_u * U)
        
        residual_mom   = self.snes_mom.getFunction()[0].norm() / denominator 
        return residual_mom
                        
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
    
    xpress = solver.Xpress
    xmass  = solver.Xmass
    xmom   = solver.Xmom
    return xpress, xmass, xmom, final_dt