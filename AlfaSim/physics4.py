import numpy as np
from models import density_model, viscosity_model, andreussi_gas_liquid,\
    colebrook_white_explicit_friction_factor, GRAVITY_CONSTANT,\
    computeGeometricProperties

'''
IN THIS CODE U IS THE SUPERFICIAL VELOCITY:
    U = u * α
'''

def calculate_residualαUPsimple(dt, UT, dtUT, αT, dtαT, P, dtP, dx, nx, dof, Mpresc, Ppresc, ρrefT, D, DhT=None, SwT=None, Si=None, H=None, fi=None):
    f = np.zeros((nx, dof))    
    
    nphases = αT.shape[1] 
                      
    A = 0.25 * np.pi * D ** 2 # [m]
    ΔV = A * dx
    
    ρg = density_model[0](P*1e5)    

    
    αG = αT[:, 0]                 
    αL = αT[:, 1]  
    αTotal = αG + αL    
    αT = np.zeros((nx, nphases))   
#     αT[:, 0] = 1 - αL / (αG + αL)
#     αT[:, 1] = 1 - αG / (αG + αL)
    αT[:, 0] = αG / αTotal
    αT[:, 1] = αL / αTotal
    
    Ur = UT[:, 0] - UT[:, 1]
        
    if H is None:
        DhT, SwT, Si, H = computeGeometricProperties(αT, D)
    
    if fi is None:
        μg = viscosity_model[0](P*1e5)
        Dhg = DhT[:, 0]    
        Rei = ρg * np.abs(Ur) * Dhg / μg #+ 1e-3

        fi = andreussi_gas_liquid(
            Rei,
            αT[:, 0],
            D,
            1e-5,
            H,
            density_model[1](P*1e5),
            ρg,
            np.abs(Ur),
            A * αT[:, 0]
        )        
        
        τi = 0.5 * fi * ρg * np.abs(Ur) * Ur   
        sign_τ = [+1, -1]
        UG = UT[:, 0]
        UL = UT[:, 1]        
        Uother = [UL, UG]
    
    for phase in range(nphases):
        
        U = UT[:, phase]
        
        α = αT[:, phase]
        
        dtU = dtUT[:, phase]
        dtα = dtαT[:, phase]
        
        ρref = ρrefT[:, phase]
        
        Dh = DhT[:, phase]
        Sw = SwT[:, phase]
        
        ρ = density_model[phase](P*1e5)
        c = density_model[phase](P*1e5, deriv=True)
        μ = viscosity_model[phase](P*1e5)
        
        ρf = 0.5 * (ρ[:-1] + ρ[1:])
        μf = 0.5 * (μ[:-1] + μ[1:])
        cf = 0.5 * (c[:-1] + c[1:])
        αf = 0.5 * (α[:-1] + α[1:])
        Sif = 0.5 * (Si[:-1] + Si[1:])
        Swf = 0.5 * (Sw[:-1] + Sw[1:])
        Dhf = 0.5 * (Dh[:-1] + Dh[1:])
        ρf = np.concatenate(([ρf[0]], ρf))
        μf = np.concatenate(([μf[0]], μf))
        cf = np.concatenate(([cf[0]], cf))
        αf = np.concatenate(([αf[0]], αf))
        Sif = np.concatenate(([Sif[0]], Sif))
        Swf = np.concatenate(([Swf[0]], Swf))
        Dhf = np.concatenate(([Dhf[0]], Dhf))
        
        Rew = ρf * np.abs(U) * Dhf / μf
    
        fw = colebrook_white_explicit_friction_factor(Rew, None, D, absolute_rugosity=1e-5)
        τw = 0.5 * fw * ρf * np.abs(U) * U          

        
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
            + α[1:-1] * ρ[1:-1] * Uc[1:  ] * A * ((0.5 - β[1:  ]) * U[2:  ] + (0.5 + β[1:  ]) * U[1:-1]) \
            - α[ :-2] * ρ[ :-2] * Uc[ :-1] * A * ((0.5 - β[ :-1]) * U[1:-1] + (0.5 + β[ :-1]) * U[ :-2]) \
            + αf[1:-1] * (P[1:-1] - P[:-2]) * 1e5 * A \
            + αf[1:-1] * ρf[1:-1] * g * np.cos(θ) * A * (H[1:-1] - H[:-2])  \
            + τw[1:-1] * (Swf[1:-1] / A) * ΔV + sign_τ[phase] * τi[1:-1] * (Sif[1:-1] / A) * ΔV
        
        # Momentum balance for half control volume
        f[-1, phase] += \
            + ρf[-1] *  U[-1] * dtαc[-1] * ΔV * 0.5 \
            + ρf[-1] * αf[-1] * dtU[-1] * ΔV * 0.5 \
            +  U[-1] * αf[-1] * c[-1] * dtPc[-1] * 1e5 * ΔV * 0.5 \
            + α[-1] * ρ[-1] * U[-1] * A * U[-1] \
            - α[-1] * ρ[-1] * Uc[-1] * A * ((0.5 - β[-1]) * U[-1] + (0.5 + β[-1]) * U[-2]) \
            + αf[-1] * (Ppresc - P[-2]) * 1e5 * A \
            + αf[-1] * ρf[-1] * g * np.cos(θ) * A * (H[-1] - H[-2])  \
            + τw[-1] * (Swf[-1] / A) * ΔV * 0.5 + sign_τ[phase] * τi[-1] * (Sif[-1] / A) * ΔV * 0.5

#         f[1:, phase] /= 1e10


   
#         f[:-1, -1] += f[:-1, phase+nphases] / ρref[:-1] - α[:-1]
        
        # TODO: Write Ap_u * u_p, i.e., find Ap_u in order to
        # create the pressure equation
        UU = np.zeros_like(f[:, phase])
        Ap_u = np.zeros_like(f[:, phase])
        # center momentum
        β = np.where(Uc > 0.0, 0.5, -0.5)
        
#             + ρf[1:-1] *  U[1:-1] * dtαc * ΔV \
#             + ρf[1:-1] * αf[1:-1] * dtU[1:-1] * ΔV \
#             +  U[1:-1] * αf[1:-1] * c[1:-1] * dtPc * 1e5 * ΔV \
            
            
        Ap_u[1:-1] = \
            + ρf[1:-1] * dtαc * ΔV \
            + ρf[1:-1] * αf[1:-1] * ΔV \
            + αf[1:-1] * c[1:-1] * dtPc * 1e5 * ΔV \
            + α[1:-1] * ρ[1:-1] * Uc[1:  ] * A * ((0.5 + β[1:  ]) ) \
            - α[ :-2] * ρ[ :-2] * Uc[ :-1] * A * ((0.5 - β[ :-1]) ) \
            + 0.5 * fw[1:-1] * ρf[1:-1] * np.abs( U[1:-1]) * (Swf[1:-1] / A) * ΔV \
            + 0.5 * fi[1:-1] * ρg[1:-1] * np.abs(Ur[1:-1]) * (Sif[1:-1] / A) * ΔV
            
        Ap_u[-1] = \
            + ρf[-1] * dtαc[-1] * ΔV \
            + ρf[-1] * αf[-1] * ΔV \
            + αf[-1] * c[-1] * dtPc[-1] * 1e5 * ΔV \
            + α[-1] * ρ[-1] * U[-1] * A \
            - α[-1] * ρ[-1] * Uc[-1] * A * ((0.5 - β[-1]) ) \
            + 0.5 * fw[-1] * ρf[-1] * np.abs( U[-1]) * (Swf[-1] / A) * ΔV * 0.5 \
            + 0.5 * fi[-1] * ρg[-1] * np.abs(Ur[-1]) * (Sif[-1] / A) * ΔV * 0.5
        
        UU[0] = Mpresc[phase] / (αf[0] * ρf[0] * A)
        UU[1:-1] = \
                   + ρf[1:-1] * αf[1:-1] * (U[1:-1] - dtU[1:-1]) * ΔV / Ap_u[1:-1] \
                   - αf[1:-1] * (P[1:-1] - P[:-2]) * 1e5 * A / Ap_u[1:-1] \
                   - αf[1:-1] * ρf[1:-1] * g * np.cos(θ) * A * (H[1:-1] - H[:-2]) / Ap_u[1:-1] \
                   - α[ :-2] * ρ[ :-2] * Uc[1:  ] * A * ((0.5 - β[1:  ]) * U[2:  ]) / Ap_u[1:-1] \
                   + α[1:-1] * ρ[1:-1] * Uc[ :-1] * A * ((0.5 + β[ :-1]) * U[ :-2]) / Ap_u[1:-1] \
                   + 0.5 * fi[1:-1] * ρg[1:-1] * np.abs(Ur[1:-1]) * Uother[phase][1:-1] * (Sif[1:-1] / A) * ΔV / Ap_u[1:-1]
                   
        UU[  -1] = \
                   + ρf[  -1] * αf[-1] * (U[-1] - dtU[-1]) * ΔV / Ap_u[  -1] \
                   - αf[  -1] * (Ppresc  - P[ -2]) * 1e5 * A / Ap_u[  -1] \
                   - αf[-1] * ρf[-1] * g * np.cos(θ) * A * (H[-1] - H[-2]) / Ap_u[  -1] \
                   - α[-1] * ρ[-1] * U[-1] * A * U[-1] / Ap_u[  -1]  \
                   + α[-1] * ρ[-1] * Uc[-1] * A * ((0.5 + β[-1]) * U[-2]) / Ap_u[  -1] \
                   + 0.5 * fi[-1] * ρg[-1] * np.abs(Ur[-1]) * Uother[phase][-1] * (Sif[-1] / A) * ΔV / Ap_u[  -1]

                       
        ρρ = np.concatenate(([ρ[0]], ρ))
        αα = np.concatenate(([α[0]], α))
        β = np.where(U > 0.0, 0.5, -0.5) 
        fP = \
            + ρ[:-1] * dtα[:-1] * ΔV \
            + α[:-1] * c[:-1] * dtP[:-1] * 1e5 * ΔV \
            + ((0.5 - β[1:  ]) * ρ[1:  ] * α[1:  ] + (0.5 + β[1:  ]) *  ρ[ :-1]  * α[ :-1]) * U[1:  ] * A \
            - ((0.5 - β[ :-1]) * ρ[ :-1] * α[ :-1] + (0.5 + β[ :-1]) * ρρ[ :-2] * αα[ :-2]) * U[ :-1] * A  
            
        f[:-1, -1] += fP
        
        ######################################
        ######################################
        # MASS CENTRAL NODES
        ρρ = np.concatenate(([ρ[0]], ρ))
        αα = np.concatenate(([α[0]], α))
        β = np.where(U > 0.0, 0.5, -0.5) 
        f[:-1, phase+nphases] +=  \
            + ρ[:-1] * dtα[:-1] * ΔV \
            + α[:-1] * c[:-1] * dtP[:-1] * 1e5 * ΔV \
            + ((0.5 - β[1:  ]) * ρ[1:  ] * α[1:  ] + (0.5 + β[1:  ]) *  ρ[ :-1]  * α[ :-1]) * U[1:  ] * A \
            - ((0.5 - β[ :-1]) * ρ[ :-1] * α[ :-1] + (0.5 + β[ :-1]) * ρρ[ :-2] * αα[ :-2]) * U[ :-1] * A 
               
        ######################################
            
        # boundaries            
        # Momentum    

        f[0,phase] = -(Mpresc[phase] - αf[0] * ρf[0] * U[0] * A)
        
        # Mass
        f[-1,phase+nphases] = -(α[-2] - α[-1])
    
#     f[:-1, -1] += 1 #  αG + αL = 1

    # pressure ghost    
    f[ -1, -1] = -(Ppresc - 0.5 * (P[-1] + P[-2]))
    
#     f[:-1, -1] = 1 - αTotal[:-1]
    
    return f

