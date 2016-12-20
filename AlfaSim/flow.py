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
        
        options = PETSc.Options()
        if options.getString('snes_type') in [self.snes.Type.VINEWTONRSLS, self.snes.Type.VINEWTONSSLS]:
            snesvi = self.snes
            
            xl = np.zeros((nx, dof))
            xl[:,:nphases] = -100
            xl[:,-1] = 0
            xl[:, 2] = 0    
            xl[:, 3] = 0    
            
            xu = np.zeros((nx, dof))    
            xu[:,:nphases] =  100
            xu[:,-1] = 100
            xu[:, 2] = 1
            xu[:, 3] = 1
                 
            xlVec = dmda.createGlobalVec()
            xuVec = dmda.createGlobalVec()   
            xlVec.setArray(xl.flatten())
            xuVec.setArray(xu.flatten())    
    
            snesvi.setVariableBounds(xlVec, xuVec)
            
        self.snes.setUpdate(self.update_function)
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
            nphases = self.nphases
            dof = self.dof
            nx  = self.nx 
            sol = self.snes.getSolution()[...]
            if len(sol) > 0:
                u = sol.reshape(nx, dof)  
                U = u[:, 0:nphases]
                α = u[:, nphases:-1]   
                αG = α[:, 0]
                αL = α[:, 1]
                UG = U[:, 0]
                UL = U[:, 1]  
                idx = UG.argmax()
                idx1 = αG.argmin()
                print('************  \n  \tΔt = %gs\t t = %gs and max α is %g' % (self.Δt, self.current_time, α.max()))
#                 print(idx, αG[idx-10:idx+1], self.H[idx-10:idx+1], UG.max(), UG[idx-1], UG[idx-4])
            else:
                print('************  \n  \tΔt = %gs\t t = %gs' % (self.Δt, self.current_time))
            max_iter = 10
            for i in range(max_iter):
                self.snes.solve(None, self.X)
                
                sol = self.snes.getSolution()[...]
                u = sol.reshape(nx, dof)
                α = u[:, nphases:-1] # view!
                αG = α[:, 0].copy()
                αL = α[:, 1].copy()
                αTotal = αG + αL
                U = u[:, 0:nphases] # view!

                αfTotal = 0.5 * (αTotal[:-1] + αTotal[1:])               
                αfTotal = np.concatenate(([αfTotal[0]], αfTotal))
        
                α[:, 0] = αG / αTotal
                α[:, 1] = αL / αTotal
                U[:, 0] *= αfTotal
                U[:, 1] *= αfTotal
#                 α[:, 0] = np.minimum(α[:, 0], 1.0)
#                 α[:, 0] = np.maximum(α[:, 0], 0.0)
#                 α[:, 1] = np.minimum(α[:, 1], 1.0)
#                 α[:, 1] = np.maximum(α[:, 1], 0.0)
#                 
#                 U[:, 0] = np.where(α[:, 0] == 0.0, U[:, 1], U[:, 0])
#                 α[:, 1] = α[:, 1] / (α[:, 0] + α[:, 1])


                if self.snes.converged:
                    break
                else:
#                     J, P, _ = self.snes.getJacobian()
#                     F = self.snes.getFunction()[0]
#                     J.zeroEntries()
#                     P.zeroEntries()
#                     F.zeroEntries()
                    self.X = self.Xold.copy()
                    self.Δt = 0.5*self.Δt
                    
            self.Xold = self.X.copy()
            self.current_time += self.Δt
            
            if self.current_time > 0.001:
                self.Δt = np.maximum(self.min_Δt, np.minimum(self.Δt*1.1, self.max_Δt))
            
    def update_function(self, snes, step):
        
        nphases = self.nphases
        
        dof = self.dof
        nx  = self.nx   
   
        sol = snes.getSolution()[...]
        u = sol.reshape(nx, dof)         
        U = u[:, 0:nphases]
        α = u[:, nphases:-1] # view!
#         αG = α[:, 0].copy()
#         αL = α[:, 1].copy()
#         αTotal = αG + αL
#         αfTotal = 0.5 * (αTotal[:-1] + αTotal[1:])               
#         αfTotal = np.concatenate(([αfTotal[0]], αfTotal))
#         α[:, 0] = αG / αTotal
#         α[:, 1] = αL / αTotal
#         U[:, 0] *= αfTotal
#         U[:, 1] *= αfTotal
#         print('max α is %g' % (α.max()))
        P = u[:, -1]
#         print('α', α.min(), α.max())
        
        # Compute ref density
        uold = self.Xold[...].reshape(nx, dof)  
        Pold = uold[:, -1]
        for phase in range(nphases):            
            self.ρref[:, phase] = density_model[phase](Pold*1e5)
            
        # Compute geometric properties
        self.Dh, self.Sw, self.Si, self.H = computeGeometricProperties(α, self.D)


#         if step > 0:
#             J,P,_ = snes.getJacobian()
#             ΔX = snes.getSolutionUpdate()
#             F = snes.getFunction()[0]
#             
#             ksp = snes.getKSP()
#             ksp.setOperators(A=J, P=P)
#             ksp.solve(b=-F, x=ΔX)
#             
#             J = J[0:nx*dof, 0:nx*dof]
#             P = P[0:nx*dof, 0:nx*dof]
#             F = F[...]
#             I = np.identity(dof*nx)
#             Δx_numpy = np.linalg.solve(J, -F)
#             Q = J.copy()
#             Q[J.diagonal() == 0, J.diagonal() == 0] = 1e-6
#             Δx_numpy = np.linalg.solve(Q, -F)
#             print('in jac', np.linalg.norm(ΔX[...] - Δx_numpy))


        
    def form_function(self, snes, X, F):       
        L   = self.L
        dof = self.dof
        nx  = self.nx
        nphases = self.nphases
        dt = self.Δt

        uold = self.Xold.getArray()
        u    = X.getArray(readonly=True)
        udot = (u - uold) / dt
        
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
    
    return x, final_dt