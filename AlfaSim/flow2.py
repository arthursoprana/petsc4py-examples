import numpy as np
from petsc4py import PETSc
from physics0 import calculate_residualαUP
from physics1 import calculate_residualαUSP
from physics3 import calculate_residualαUSPsimple
from physics4 import calculate_residualαUPsimple
from models import density_model, computeGeometricProperties
from physics5 import calculate_residual_mass, calculate_residual_mom,\
    calculate_velocity_update, calculate_jacobian_mass, calculate_coeff_mom
from scipy import sparse

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
        self.J = dmda_mass.createMatrix()
  
        self.Fmom = dmda_mom.createGlobalVec()
        self.Xmom = dmda_mom.createGlobalVec()
        self.Xmomold = dmda_mom.createGlobalVec()
        
#         options = PETSc.Options()
#         options.setValue('-mat_fd_coloring_err', 1e-3)
#         options.setValue('-mat_fd_coloring_umin', 1e-3)
#         options.setValue('-snes_linesearch_type', 'l2')
        self.snes_mom.setFunction(self.form_function_mom, self.Fmom)
        self.snes_mom.setUseFD() # Enables coloring, same as -snes_fd_color
        self.snes_mom.setFromOptions()
        
        options = PETSc.Options()
#         options.setValue('-npc_snes_type', 'nrichardson')
#         options.setValue('-mat_fd_coloring_err', 1e-5)
#         options.setValue('-mat_fd_coloring_umin', 1e-0)          
        #self.snes.setUpdate(self.update_function)
        self.snes.setFunction(self.form_function, self.F)
        self.snes.setJacobian(self.form_jacobian, self.J)
#         self.snes.setUseFD() # Enables coloring, same as -snes_fd_color
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
        u = initial_solution.reshape(self.nx, self.dof)        
        U = u[:, 0:self.nphases]
        α = u[:, self.nphases:-1]
        P = u[:, -1]
        
        self.Pprev = P.copy()
        
        initial_solution_mass = np.hstack((α, P[:, np.newaxis]))
        self.X[...] = initial_solution_mass.flatten()
        self.Xold[...] = self.X[...].copy()
        
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
#         ksp_mom = self.snes_mom.getKSP()
#         ksp_mom.setType(ksp_mom.Type.PREONLY)
#         pc_mom = ksp_mom.getPC()
#         self.snes.setType(self.snes.Type.KSPONLY)
        # CHUNCHO!
#         options = PETSc.Options()
#         options.setValue('-snes_linesearch_damping', 0.99)
#         self.snes_mom.setFromOptions()
#         options = PETSc.Options()
#         options.setValue('-snes_type', 'composite')
#         options.setValue('-snes_composite_type', 'additiveoptimal')
#         options.setValue('-snes_composite_sneses', 'newtonls,fas')
#         options.setValue('-snes_composite_damping', (1,1))
#         options.setValue('-sub_0_snes_linesearch_type', 'basic')
#         self.snes.setFromOptions()
#         
        self.snes_mom.setTolerances(rtol=1e-50, stol=1e-50, atol=1e-6, max_it=3)
        self.snes.setTolerances(rtol=1e-50, stol=1e-50, atol=1e-7, max_it=20)
        
        while self.current_time < self.final_time:
            print('************  \n  \tΔt = %gs\t t = %gs' % (self.Δt, self.current_time))
            max_breaks = 10
            max_iter = 10

            for i in range(max_breaks):
                
                for j in range(max_iter):
                    print('Solve mom')
                    self.snes_mom.solve(None, self.Xmom)                     
                    self.calc_coeff_mom()                   
                    
                    print('Solve mass')
                    u    = self.Xmom.getArray()                  
                    self.Uprev = u.reshape(nx, nphases).copy()
                    
                    sol = self.X[...]
                    u = sol.reshape(nx, dof-nphases)
                    self.αprev = u[:, :-1].copy()
                    self.Pprev = u[:, -1].copy()
                    PprevLocal = u[:, -1].copy()
                    self.snes.solve(None, self.X)

                    self.update_velocities_estimates(PprevLocal)    
                    
                    sol = self.X[...]
                    u = sol.reshape(nx, dof-nphases)
                    α = u[:, :-1] # view!
                    P = u[:, -1]
                    ΔP = P - PprevLocal
                    normΔP = np.linalg.norm(ΔP)
                    print(' \t\t %i ->>>>>ΔP norm is %g' % (j, normΔP))
                    
#                     assert False
                    
                    if normΔP < 1e-6:
                        break
                    
                    if np.any(α < 0.0) or np.any(α > 1.0):
                        break
#                     if self.snes.diverged:
#                         α[:] = αprev.copy()
#                         P[:] = Pprev.copy()       
                    
#                 if self.snes.converged and self.snes_mom.converged:
#                 if np.any(α < 0.0) or np.any(α > 1.0):
                if self.snes.converged:
                    break
                else:
                    print('\t\t ******* BREAKING TIMESTEP %i *******' % i)
                    assert False
                    self.X = self.Xold.copy()
                    self.Xmom = self.Xmomold.copy()
                    self.Δt = 0.5*self.Δt
                    
            self.Xold = self.X.copy()
            self.Xmomold = self.Xmom.copy()
            self.current_time += self.Δt
            
            # Update ρref
            uold = self.Xold[...].reshape(nx, dof-nphases)
            Pold = uold[:, -1]
            for phase in range(nphases):            
                self.ρref[:, phase] = density_model[phase](Pold*1e5)
            
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
#         αG = α[:, 0].copy()                 
#         αL = α[:, 1].copy()                 
#         α[:, 0] = αG / (αG + αL)
#         α[:, 1] = αL / (αG + αL)
        
#         print(α)
#         print(α.sum(axis=1))
#         print(P)
        # Compute ref density
        uold = self.Xold[...].reshape(nx, dof-nphases)  
        Pold = uold[:, -1]
        for phase in range(nphases):            
            self.ρref[:, phase] = density_model[phase](Pold*1e5)
            
        # Compute geometric properties
        self.Dh, self.Sw, self.Si, self.H = computeGeometricProperties(α, self.D)
        
        uold = self.Xold.getArray()     
        uold = uold.reshape(nx, dof-nphases)        
        αold = uold[:, :-1]
        Pold = uold[:,  -1]
                    
        # Correct velocities
        L   = self.L
        dx = L / (nx - 1)
        dt = self.Δt
        uold = self.Xmomold.getArray()     
        uold = uold.reshape(nx, nphases)        
        Uold = uold.copy()
        u    = self.Xmom.getArray()                  
        U = u.reshape(nx, nphases) # view!
        ΔP = P - self.Pprev
        #ΔU = calculate_velocity_update(ΔP, None, dt, U, Uold, α, αold, P, Pold, dx, nx, dof, self.Mpresc, self.Ppresc, self.ρref, self.D)
        #U[:] = U + ΔU
        self.Pprev = P.copy()
        
    def form_function(self, snes, X, F):      
        
        F[...] = 0.0 # For safety

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
        
        #self.Pprev = P.copy()
        
        # getting from mom
        uold = self.Xmomold.getArray()
        u    = self.Xmom.getArray()       
        uold = uold.reshape(nx, nphases)        
        Uold = uold.copy()              
        U = u.reshape(nx, nphases) # view!
        
        dx = L / (nx - 1)
        
        Pprev = self.Pprev
        
        if not self.snes.use_fd and self.snes.getIterationNumber() > 0:
            Δu = self.snes.getSolutionUpdate()[...]
            Δu = Δu.reshape(nx, dof-nphases)            
            Δα = Δu[:, :-1]
            ΔP = Δu[:,  -1]        
            #ΔU = calculate_velocity_update(ΔP, None, dt, U, Uold, α, αold, P, Pold, dx, nx, dof, self.Mpresc, self.Ppresc, self.ρref, self.D, Ap_uT=self.Ap_u)
            #U[:] = U + ΔU
            
        residual = calculate_residual_mass(dt, U, Uold, α, αold, P, Pold, dx, nx, dof, self.Mpresc, self.Ppresc, 
                                      self.ρref, self.D, self.Dh, self.Sw, self.Si, None, None, Pprev, self.Ap_u)
        F.setArray(residual.flatten())

    def form_jacobian(self, snes, X, J, P): 
        
        J.zeroEntries()
        
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
        u    = self.Xmom.getArray()       
        uold = uold.reshape(nx, nphases)        
        Uold = uold.copy()              
        U = u.reshape(nx, nphases) # view!
        
        dx = L / (nx - 1)
        
        Pprev = self.Pprev
        
        if not self.snes.use_fd:
            ΔP = P - Pprev
            #ΔU = calculate_velocity_update(ΔP, None, dt, U, Uold, α, αold, P, Pold, dx, nx, dof, self.Mpresc, self.Ppresc, self.ρref, self.D, Ap_uT=self.Ap_u)
            #U[:] = U + ΔU

        row, col, data = calculate_jacobian_mass(dt, U, Uold, α, αold, P, Pold, dx, nx, dof, self.Mpresc, self.Ppresc, 
                                      self.ρref, self.D, self.Dh, self.Sw, self.Si, None, None, Pprev, self.Ap_u)


        size = (dof-nphases)*nx
        jac = sparse.coo_matrix((data, (row, col)), shape=(size, size))
        jac_csr = jac.tocsr()
        J.setValuesCSR(I=jac_csr.indptr, J=jac_csr.indices, V=jac_csr.data)     
        J.assemblyBegin()
        J.assemblyEnd()
        
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
        
        # getting from mass
        uold = self.Xold.getArray()
        u    = self.X.getArray(readonly=True)      
        uold = uold.reshape(nx, dof-nphases)        
        αold = uold[:, :-1]
        Pold = uold[:,  -1]        
        u = u.reshape(nx, dof-nphases)        
        α = u[:, :-1]
        P = u[:,  -1]
        
        calculate_coeff_mom(dt, U, Uold, α, αold, P, Pold, dx, nx, dof, self.Mpresc, self.Ppresc, 
             self.ρref, self.D, self.Dh, self.Sw, self.Si, None, None, self.Ap_u) 

    def update_velocities_estimates(self, Pprev):        
        L   = self.L
        dof = self.dof
        nx  = self.nx
        nphases = self.nphases
        dt = self.Δt  
        dx = L / (nx - 1)
    
        uold = self.Xold.getArray()     
        uold = uold.reshape(nx, dof-nphases)        
        αold = uold[:, :-1]
        Pold = uold[:,  -1]
        
        uold = self.Xmomold.getArray()     
        uold = uold.reshape(nx, nphases)        
        Uold = uold.copy()        
    
        u    = self.Xmom.getArray()                  
        U = u.reshape(nx, nphases) # view!
        
        sol = self.X[...]
        u = sol.reshape(nx, dof-nphases)
        α = u[:, :-1] # view!
        P = u[:, -1]
        
        Δα = α - self.αprev
        ΔP = P - Pprev
    
        ΔU = calculate_velocity_update(ΔP, Δα, dt, U, Uold, α, αold, P, Pold, dx, nx, dof, self.Mpresc, self.Ppresc, self.ρref, self.D, Ap_uT=self.Ap_u)
        U[:] = self.Uprev + ΔU   
                       
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