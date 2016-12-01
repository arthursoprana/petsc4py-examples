import numpy as np
from models import density_model, viscosity_model, andreussi_gas_liquid,\
    colebrook_white_explicit_friction_factor, GRAVITY_CONSTANT,\
    computeGeometricProperties


def calculate_residual(dt, UT, dtUT, αT, dtαT, P, dtP, dx, nx, dof, ρrefT, D, DhT=None, SwT=None, Si=None, H=None, fi=None):
    f = np.zeros((nx, dof))    
    
    nphases = αT.shape[1] 
    
    Ppresc  = 1.0 # [bar]
    USpresc = [3.0, 0.6] # [m/s]
    
                      
    A = 0.25 * np.pi * D ** 2 # [m]
    ΔV = A * dx
    
    ρg = density_model[0](P*1e5)
    
    αT = np.maximum(αT, 0.0)
    αT = np.minimum(αT, 1.0)
    
    if H is None:
        DhT, SwT, Si, H = computeGeometricProperties(αT, D)
    
    if fi is None:
        μg = viscosity_model[0](P*1e5)
        Dhg = DhT[:, 0]    
        Ur = np.abs(UT[:, 0] - UT[:, 1])
        Rei = ρg * np.abs(Ur) * Dhg / μg + 1e-3

        fi = andreussi_gas_liquid(
            Rei,
            αT[:, 0],
            D,
            1e-5,
            H,
            density_model[1](P*1e5),
            ρg,
            Ur,
            A * αT[:, 0]
        )        

    
    
    for phase in range(nphases):
        
        U = UT[:, phase]
        α = αT[:, phase]
        
        dtU = dtUT[:, phase]
        dtα = dtαT[:, phase]
        
        ρref = ρrefT[:, phase]
        
        Dh = DhT[:, phase]
        Sw = SwT[:, phase]
        
        ρ = density_model[phase](P*1e5)
        c = density_model[phase](P, deriv=True)
        μ = viscosity_model[phase](P*1e5)
        
        ρf = 0.5 * (ρ[:-1] + ρ[1:])
        cf = 0.5 * (c[:-1] + c[1:])
        αf = 0.5 * (α[:-1] + α[1:])
        Sif = 0.5 * (Si[:-1] + Si[1:])
        Swf = 0.5 * (Sw[:-1] + Sw[1:])
        Dhf = 0.5 * (Dh[:-1] + Dh[1:])
        ρf = np.concatenate(([ρf[0]], ρf))
        cf = np.concatenate(([cf[0]], cf))
        αf = np.concatenate(([αf[0]], αf))
        Sif = np.concatenate(([Sif[0]], Sif))
        Swf = np.concatenate(([Swf[0]], Swf))
        Dhf = np.concatenate(([Dhf[0]], Dhf))
        
        Rew = ρ * np.abs(U) * Dhf / μ
    
        fw = colebrook_white_explicit_friction_factor(Rew, None, D, absolute_rugosity=1e-5)
        τw = 0.5 * fw * ρf * np.abs(U) * U          
        
        Ur = U - np.take(UT, phase+1, axis=1, mode='wrap')
        τi = 0.5 * fi * ρg * np.abs(Ur) * Ur    
        
        ######################################
        # MOMENTUM CENTRAL NODES
        # Staggered
        Uc = 0.5 * (U[1:] + U[:-1])
        dtPc = 0.5 * (dtP[:-2] + dtP[1:-1])
        dtαc = 0.5 * (dtα[:-2] + dtα[1:-1])
        
        θ = 0.0 # for now
        g = GRAVITY_CONSTANT 
           
        β = np.where(Uc > 0.0, 0.5, -0.5)
        # center momentum
        f[1:-1, phase] += \
            + ρf[1:-1] *  U[1:-1] * dtαc * ΔV \
            + ρf[1:-1] * αf[1:-1] * dtU[1:-1] * ΔV \
            +  U[1:-1] * αf[1:-1] * c[1:-1] * dtPc * 1e5 * ΔV \
            + α[ :-2] * ρ[ :-2] * Uc[1:  ] * A * ((β[1:  ] - 0.5) * U[2:  ] + (β[1:  ] + 0.5) * U[1:-1]) \
            - α[1:-1] * ρ[1:-1] * Uc[ :-1] * A * ((β[ :-1] - 0.5) * U[1:-1] + (β[ :-1] + 0.5) * U[ :-2]) \
            + αf[1:-1] * (P[1:-1] - P[:-2]) * 1e5 * A \
            + αf[1:-1] * ρf[1:-1] * g * np.cos(θ) * A * (H[1:-1] - H[:-2])  \
            + τw[1:-1] * (Swf[1:-1] / A) * ΔV + τi[1:-1] * (Sif[1:-1] / A) * ΔV
        
        # Momentum balance for half control volume
        f[-1, phase] += \
            + ρf[-1] *  U[-1] * dtαc[-1] * ΔV \
            + ρf[-1] * αf[-1] * dtU[-1] * ΔV \
            +  U[-1] * αf[-1] * c[-1] * dtPc[-1] * 1e5 * ΔV \
            + α[-1] * ρ[-1] * U[-1] * A * U[-1] \
            - α[-1] * ρ[-1] * Uc[-1] * A * ((β[-1] - 0.5) * U[-1] + (β[-1] + 0.5) * U[-2]) \
            + αf[-1] * (Ppresc - P[-2]) * 1e5 * A \
            + αf[-1] * ρf[-1] * g * np.cos(θ) * A * (H[-1] - H[-2])  \
            + τw[-1] * (Swf[-1] / A) * ΔV + τi[-1] * (Sif[-1] / A) * ΔV

        f[1:, phase] /= USpresc[phase] * ρref[:-1] #* 1e6 #/ dx
#         f[1:, phase] /= USpresc[phase] * ρf[1:] * 1e6
        
#         f[1:, phase] /= ρref[:-1] * 1000 / dt
        ######################################
        ######################################
        # MASS CENTRAL NODES
        ρρ = np.concatenate(([ρ[0]], ρ))
        αα = np.concatenate(([α[0]], α))
        β = np.where(U > 0.0, 0.5, -0.5) 
        f[:-1, phase+nphases] +=  \
            + ρ[:-1] * dtα[:-1] * ΔV \
            + α[:-1] * c[:-1] * dtP[:-1] * 1e5 * ΔV \
            + ((β[1:  ] - 0.5) * ρ[1:  ] * α[1:  ] + (β[1:  ] + 0.5) *  ρ[ :-1]  * α[ :-1]) * U[1:  ] * A \
            - ((β[ :-1] - 0.5) * ρ[ :-1] * α[ :-1] + (β[ :-1] + 0.5) * ρρ[ :-2] * αα[ :-2]) * U[ :-1] * A  
               
        ######################################
   
        f[:-1, -1] += f[:-1, phase+nphases] / ρref[:-1] - α[:-1]
#         f[:-1, -1] += f[:-1, phase+nphases] / ρ[:-1] - α[:-1]
        
        # boundaries            
        # Momentum            
        if αf[0] < 1e-3: # Fix for low α value
            f[ 0,phase] = -(USpresc[phase] - U[0] * 0.001)
        else:            
            f[ 0,phase] = -(USpresc[phase] - U[0] * αf[0])
        
        # Mass
        f[-1,phase+nphases] = -(α[-2] - α[-1])
    
    f[:-1, -1] += 1  #  αG + αL = 1

    # pressure ghost    
    f[ -1, -1] = -(Ppresc - 0.5 * (P[-1] + P[-2]))
#     f[ -1, -1] = Ppresc -  P[-1]
    
    return f

class Flow0(object):
    def __init__(self, dm, nx, dof, pipe_length, nphases, α0):
        self.dm  = dm        
        self.L   = pipe_length
        self.nx  = nx
        self.dof = dof
        self.nphases = nphases
        self.D =  0.1 # [m]   
        self.ρref = np.zeros((nx, nphases))
        for phase in range(nphases):
            self.ρref[:, phase] = density_model[phase](1e5)

        self.Dh, self.Sw, self.Si, self.H = computeGeometricProperties(α0, self.D)
        
        #self.fi = np.zeros(nx)
    
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

#         sol = snes.getSolution()[...]
#         u = sol.reshape(nx, dof)
#          
#         U = u[:, 0:nphases]
#         α = u[:, nphases:-1]
#         P = u[:, -1]
#  
#         u[:, nphases:-1] = np.where(u[:, nphases:-1] < 0.0, 0.0, u[:, nphases:-1])
#         u[:, nphases:-1] = np.where(u[:, nphases:-1] > 1.0, 1.0, u[:, nphases:-1])
          
#         αG = α[:, 0]
#         αL = α[:, 1]
#          
#         u[:, 2] = αG / (αG + αL)
#         u[:, 3] = αL / (αG + αL)        
#         snes.getSolution()[...] = u.flatten()
#         
#         ρg = density_model[0](P*1e5)
#         μg = viscosity_model[0](P*1e5)
#         Dhg = self.Dh[:, 0]
#     
#         Ur = np.abs(U[:, 0] - U[:, 1])
#         Rei = ρg * np.abs(Ur) * Dhg / μg + 1e-3
#         D = self.D
#         A = 0.25 * np.pi * D ** 2 
#         fi = andreussi_gas_liquid(
#             Rei,
#             α[:, 0],
#             D,
#             1e-5,
#             self.H,
#             density_model[1](P*1e5),
#             ρg,
#             Ur,
#             A * α[:, 0]
#         )
# 
#         self.fi = fi

        
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
            
                        residual = calculate_residual(dt, U, dtU, α, dtα, P, dtP, dx, nx, dof, self.ρref, 
                                                      self.D, self.Dh, self.Sw, self.Si, self.H, None)
                        F.setArray(residual.flatten())