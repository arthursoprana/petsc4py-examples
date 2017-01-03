import numpy as np
from models import density_model, viscosity_model, andreussi_gas_liquid,\
    colebrook_white_explicit_friction_factor, GRAVITY_CONSTANT,\
    computeGeometricProperties


def calculate_residualαUP(dt, UT, dtUT, αT, dtαT, P, dtP, dx, nx, dof, Mpresc, Ppresc, ρrefT, D, DhT=None, SwT=None, Si=None, H=None, fi=None):
    f = np.zeros((nx, dof))    
    
    nphases = αT.shape[1] 
                      
    A = 0.25 * np.pi * D ** 2 # [m]
    ΔV = A * dx
    g = GRAVITY_CONSTANT 
    
    αG = αT[:, 0]                 
    αL = αT[:, 1]  
    αTotal = αG + αL 
    
#     αT = np.zeros((nx, nphases))   
#  
#     αT[:, 0] = αG / (αG + αL)
#     αT[:, 1] = αL / (αG + αL)    
     
    αGf = 0.5 * (αG[:-1] + αG[1:])   
    αLf = 0.5 * (αL[:-1] + αL[1:])   
    αGf = np.concatenate(([αGf[0]], αGf))
    αLf = np.concatenate(([αLf[0]], αLf))
           
#     UG = UT[:, 0]                 
#     UL = UT[:, 1]  
#     UT = np.zeros((nx, nphases))   
#     UT[:, 0] = UG * (αGf + αLf)
#     UT[:, 1] = UL * (αGf + αLf)
    
    ρg = density_model[0](P*1e5)
    ρL = density_model[1](P*1e5)
    
    ρm = ρg * αT[:, 0] + ρL * αT[:, 1]
    αG = αT[:, 0].copy()
    αL = αT[:, 1].copy()

    Ur = UT[:, 0] - UT[:, 1]
    γ = 1.2 # suggested value
    γ = 0.0
    Pd = γ * ((αL * ρL * αG * ρg) / ρm / αTotal ** 2) * Ur ** 2 
#     Pd = αG * αL * (ρL - ρg) * g * D

    
    ρG = density_model[0](P*1e5)
    ρGf = 0.5 * (ρG[:-1] + ρG[1:])
    ρGf = np.concatenate(([ρGf[0]], ρGf))
    ρL = density_model[1](P*1e5)
    ρLf = 0.5 * (ρL[:-1] + ρL[1:])
    ρLf = np.concatenate(([ρLf[0]], ρLf))
    
    if H is None:
        DhT, SwT, Si, H = computeGeometricProperties(αT, D)
    
    
    if fi is None:
        μG = viscosity_model[0](P*1e5)
        DhG = DhT[:, 0]        
        
        μGf  = 0.5 * (μG[:-1] + μG[1:])        
        αGf  = 0.5 * (αG[:-1] + αG[1:])        
        DhGf = 0.5 * (DhG[:-1] + DhG[1:])        
        Hf = 0.5 * (H[:-1] + H[1:])        
        μGf  = np.concatenate(([μGf[0]], μGf))        
        αGf  = np.concatenate(([αGf[0]], αGf))       
        DhGf = np.concatenate(([DhGf[0]], DhGf))
        Hf = np.concatenate(([Hf[0]], Hf))
        
        Rei = ρGf * np.abs(Ur) * DhGf / μGf

        fi = andreussi_gas_liquid(
            Rei,
            αGf,
            D,
            1e-5,
            Hf,
            ρLf,
            ρGf,
            np.abs(Ur),
            A * αGf
        )        
        
    τi = 0.5 * fi * ρGf * np.abs(Ur) * Ur   
    sign_τ = [+1, -1]

    
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
        μf = 0.5 * (μ[:-1] + μ[1:])
        cf = 0.5 * (c[:-1] + c[1:])
        αf = 0.5 * (α[:-1] + α[1:])
        Sif = 0.5 * (Si[:-1] + Si[1:])
        Swf = 0.5 * (Sw[:-1] + Sw[1:])
        Dhf = 0.5 * (Dh[:-1] + Dh[1:])
        
        # Harmonic mean
#         αf  = 2.0 * ( α[:-1] *  α[1:]) / ( α[:-1] +  α[1:])
#         Sif = 2.0 * (Si[:-1] * Si[1:]) / (Si[:-1] + Si[1:])
#         Swf = 2.0 * (Sw[:-1] * Sw[1:]) / (Sw[:-1] + Sw[1:])
#         Dhf = 2.0 * (Dh[:-1] * Dh[1:]) / (Dh[:-1] + Dh[1:])

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
        ######################################
        # MASS FLUXES
        ρρ = np.concatenate(([ρ[0]], ρ))
        αα = np.concatenate(([10], α))
        β = np.where(U > 0.0, 0.5, -0.5) 
        
        me = ((0.5 - β[1:  ]) * ρ[1:  ] * α[1:  ] + (0.5 + β[1:  ]) *  ρ[ :-1]  * α[ :-1]) * U[1:  ] * A
        mw = ((0.5 - β[ :-1]) * ρ[ :-1] * α[ :-1] + (0.5 + β[ :-1]) * ρρ[ :-2] * αα[ :-2]) * U[ :-1] * A  
 
        ######################################
        
        ######################################
        # MOMENTUM CENTRAL NODES
        # Staggered
        Uc = 0.5 * (U[1:] + U[:-1])

#         Uc = (1 / α[:-1]) * 2.0 * (αf[1:] * U[1:] * αf[:-1] * U[:-1]) / (αf[1:] * U[1:] + αf[:-1] * U[:-1])
        
        mec = 0.5 * (me[1:] + me[:-1])
        mwc = 0.5 * (mw[1:] + mw[:-1])
        dtPc = 0.5 * (dtP[:-2] + dtP[1:-1])
        dtαc = 0.5 * (dtα[:-2] + dtα[1:-1])

        
        θ = 0.0 # for now        
        
        αfpress = αf.copy()
        αfpress = np.where(αfpress < 1e-3, 0.0, αfpress)

        
        β = np.where(Uc > 0.0, 0.5, -0.5)
        # center momentum
#         f[1:-1, phase] += \
#             + ρf[1:-1] *  U[1:-1] * dtαc * ΔV \
#             + ρf[1:-1] * αf[1:-1] * dtU[1:-1] * ΔV \
#             +  U[1:-1] * αf[1:-1] * c[1:-1] * dtPc * 1e5 * ΔV \
#             + mec * ((0.5 - β[1:  ]) * U[2:  ] + (0.5 + β[1:  ]) * U[1:-1]) \
#             - mwc * ((0.5 - β[ :-1]) * U[1:-1] + (0.5 + β[ :-1]) * U[ :-2]) \
#             + αf[1:-1] * (P[1:-1] - P[:-2]) * 1e5 * A \
#             + αf[1:-1] * ρf[1:-1] * g * np.cos(θ) * A * (H[1:-1] - H[:-2]) \
#             + τw[1:-1] * (Swf[1:-1] / A) * ΔV + sign_τ[phase] * τi[1:-1] * (Sif[1:-1] / A) * ΔV \
#             + αf[1:-1] * (Pd[1:-1] - Pd[:-2]) * A
        
        # Momentum balance for half control volume
#         f[-1, phase] += \
#             + ρf[-1] *  U[-1] * dtαc[-1] * ΔV * 0.5 \
#             + ρf[-1] * αf[-1] * dtU[-1] * ΔV * 0.5 \
#             +  U[-1] * αf[-1] * c[-1] * dtPc[-1] * 1e5 * ΔV * 0.5 \
#             + me[-1] * U[-1] \
#             - me[-1] * ((0.5 - β[-1]) * U[-1] + (0.5 + β[-1]) * U[-2]) \
#             + αf[-1] * (Ppresc - P[-2]) * 1e5 * A \
#             + αf[-1] * ρf[-1] * g * np.cos(θ) * A * (H[-1] - H[-2])  \
#             + τw[-1] * (Swf[-1] / A) * ΔV * 0.5 + sign_τ[phase] * τi[-1] * (Sif[-1] / A) * ΔV * 0.5 \
#             + αf[-1] * (Pd[-1] - Pd[-2]) * A
        f[1:-1, phase] += \
            + ρf[1:-1] *  U[1:-1] * dtαc * ΔV \
            + ρf[1:-1] * αf[1:-1] * dtU[1:-1] * ΔV \
            +  U[1:-1] * αf[1:-1] * c[1:-1] * dtPc * 1e5 * ΔV \
            + α[1:-1] * ρ[1:-1] * Uc[1:  ] * A * ((0.5 - β[1:  ]) * U[2:  ] + (0.5 + β[1:  ]) * U[1:-1]) \
            - α[ :-2] * ρ[ :-2] * Uc[ :-1] * A * ((0.5 - β[ :-1]) * U[1:-1] + (0.5 + β[ :-1]) * U[ :-2]) \
            + αf[1:-1] * (P[1:-1] - P[:-2]) * 1e5 * A \
            + αf[1:-1] * ρf[1:-1] * g * np.cos(θ) * A * (H[1:-1] - H[:-2])  \
            + τw[1:-1] * (Swf[1:-1] / A) * ΔV + sign_τ[phase] * τi[1:-1] * (Sif[1:-1] / A) * ΔV \
            + αf[1:-1] * (Pd[1:-1] - Pd[:-2]) * A
        f[-1, phase] += \
            + ρf[-1] *  U[-1] * dtαc[-1] * ΔV * 0.5 \
            + ρf[-1] * αf[-1] * dtU[-1] * ΔV * 0.5 \
            +  U[-1] * αf[-1] * c[-1] * dtPc[-1] * 1e5 * ΔV * 0.5 \
            + α[-1] * ρ[-1] * U[-1] * A * U[-1] \
            - α[-1] * ρ[-1] * Uc[-1] * A * ((0.5 - β[-1]) * U[-1] + (0.5 + β[-1]) * U[-2]) \
            + αf[-1] * (Ppresc - P[-2]) * 1e5 * A \
            + αf[-1] * ρf[-1] * g * np.cos(θ) * A * (H[-1] - H[-2])  \
            + τw[-1] * (Swf[-1] / A) * ΔV * 0.5 + sign_τ[phase] * τi[-1] * (Sif[-1] / A) * ΔV * 0.5 \
            + αf[-1] * (Pd[-1] - Pd[-2]) * A
        
#         f[1:, phase] /= 1e5
        
        ######################################
        ######################################
        # MASS CENTRAL NODES
        ρρ = np.concatenate(([ρ[0]], ρ))
        αα = np.concatenate(([1], α))
        β = np.where(U > 0.0, 0.5, -0.5) 
        
        
        fp =  \
            + ρ[:-1] * dtα[:-1] * ΔV \
            + α[:-1] * c[:-1] * dtP[:-1] * 1e5 * ΔV \
            + ((0.5 - β[1:  ]) * ρ[1:  ] * α[1:  ] + (0.5 + β[1:  ]) *  ρ[ :-1]  * α[ :-1]) * U[1:  ] * A \
            - ((0.5 - β[ :-1]) * ρ[ :-1] * α[ :-1] + (0.5 + β[ :-1]) * ρρ[ :-2] * αα[ :-2]) * U[ :-1] * A  
        f[:-1, phase+nphases] += fp
#         f[:-1, -1] += fp
        ######################################
   
#         f[:-1, -1] += f[:-1, phase+nphases]
#         f[:-1, -1] += f[:-1, phase+nphases] - α[:-1]
#         f[:-1, -1] += f[:-1, phase+nphases] / ρref[:-1] - α[:-1]
#         f[:-1, -1] += dt * f[:-1, phase+nphases] / ρ[:-1] - α[:-1]
#         f[:-1, -1] += - α[:-1]
        
        # boundaries            
        # Momentum            
        f[0,phase] = -(Mpresc[phase] - αα[0] * ρρ[0] * U[0] * A)
        
        # Mass
        f[-1,phase+nphases] = -(α[-2] - α[-1])
    
    f[:-1, -1] = 1 - αTotal[:-1]

    # pressure ghost    
    f[ -1, -1] = -(Ppresc - P[-1])


    return f
