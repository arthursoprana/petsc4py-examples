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
    
    αT = np.zeros((nx, nphases))
   
    αT[:, 0] = αG / (αG + αL)
    αT[:, 1] = αL / (αG + αL)    
      
    αGf = 0.5 * (αG[:-1] + αG[1:])   
    αLf = 0.5 * (αL[:-1] + αL[1:])   
    αGf = np.concatenate(([αGf[0]], αGf))
    αLf = np.concatenate(([αLf[0]], αLf))
            
    UG = UT[:, 0]                 
    UL = UT[:, 1]  
    UT = np.zeros((nx, nphases))   
    UT[:, 0] = UG * (αGf + αLf)
    UT[:, 1] = UL * (αGf + αLf)
        
    ρG = density_model[0](P*1e5)
    ρL = density_model[1](P*1e5)
    ρ = np.array([ρG, ρL]).T
    
    ρGf = 0.5 * (ρG[:-1] + ρG[1:])
    ρLf = 0.5 * (ρL[:-1] + ρL[1:])
    
    ρGf = np.concatenate(([ρGf[0]], ρGf))
    ρLf = np.concatenate(([ρLf[0]], ρLf))

    sound_speed = [316, 14354.4]
    
    ρm = ρG * αT[:, 0] + ρL * αT[:, 1]
    αG = αT[:, 0]
    αL = αT[:, 1]

    Ur = UT[:, 0] - UT[:, 1]
    γ = 1.2 # suggested value
    γ = 0.0
#     Pd = γ * ((αL * ρL * αG * ρG) / ρm / αTotal ** 2) * Ur ** 2 
    Pd = αG * αL * (ρL - ρG) * g * D
    
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
    sign_τ = np.array([+1, -1])
    
    ######################################
    ######################################
    # BEGIN PHASE "LOOP" 
    ######################################
    ######################################
    U = UT
    α = αT
    
    dtU = dtUT
    dtα = dtαT
    
    ρref = ρrefT
    
    Dh = DhT
    Sw = SwT
    
    cG = density_model[0](P*1e5, deriv=True)
    cL = density_model[1](P*1e5, deriv=True)
    c = np.array([cG, cL]).T
    
    μG = viscosity_model[0](P*1e5)
    μL = viscosity_model[1](P*1e5)
    μ = np.array([μG, μL]).T
    
    ρf = 0.5 * (ρ[:-1] + ρ[1:])
    μf = 0.5 * (μ[:-1] + μ[1:])
    cf = 0.5 * (c[:-1] + c[1:])
    ρreff = 0.5 * (ρref[:-1] + ρref[1:])
    
    # Arithmetic mean
    αf = 0.5 * (α[:-1] + α[1:])
    Sif = 0.5 * (Si[:-1] + Si[1:])
    Swf = 0.5 * (Sw[:-1] + Sw[1:])
    Dhf = 0.5 * (Dh[:-1] + Dh[1:])
#         # Harmonic mean
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
    ρreff = np.concatenate(([ρreff[0]], ρreff))

    # Arithmetic mean
    Uc = 0.5 * (U[1:] + U[:-1])
#         # Harmonic mean
#         Uc = (1 / α[:-1]) * 2.0 * (αf[1:] * U[1:] * αf[:-1] * U[:-1]) / (αf[1:] * U[1:] + αf[:-1] * U[:-1])
    
    Rew = ρf * np.abs(U) * Dhf / μf

    fw = colebrook_white_explicit_friction_factor(Rew, None, D, absolute_rugosity=1e-5)
    τw = 0.5 * fw * ρf * np.abs(U) * U 
    
    ######################################
    ######################################
    # MASS FLUXES
    ρρ = np.concatenate(([ρ[0]], ρ))
    αα = np.concatenate(([α[0]], α))
    β = np.where(U > 0.0, 0.5, -0.5)
    
    me = ((0.5 - β[1:  ]) * ρ[1:  ] * α[1:  ] + (0.5 + β[1:  ]) *  ρ[ :-1]  * α[ :-1]) * U[1:  ] * A
    mw = ((0.5 - β[ :-1]) * ρ[ :-1] * α[ :-1] + (0.5 + β[ :-1]) * ρρ[ :-2] * αα[ :-2]) * U[ :-1] * A  
    
    # Left BC
    mw[0] = Mpresc
    
#         mec = 0.5 * (me[1:] + me[:-1])
#         mwc = 0.5 * (mw[1:] + mw[:-1])        
    mec = α[1:-1] * ρ[1:-1] * Uc[1:  ] * A
    mwc = α[ :-2] * ρ[ :-2] * Uc[ :-1] * A
    
    ######################################
    
    ######################################
    # MOMENTUM CENTRAL NODES
    # Staggered
    dtPc = 0.5 * (dtP[:-2] + dtP[1:-1])
    dtαc = 0.5 * (dtα[:-2] + dtα[1:-1])
    θ = 0 # for now
    β = np.where(Uc > 0.0, 0.5, -0.5)
    # center momentum
    f[1:-1,:nphases] += \
        + ρf[1:-1] *  U[1:-1] * dtαc * ΔV \
        + ρf[1:-1] * αf[1:-1] * dtU[1:-1] * ΔV \
        +  U[1:-1] * αf[1:-1] * c[1:-1] * dtPc[:, np.newaxis] * 1e5 * ΔV \
        + mec * ((0.5 - β[1:  ]) * U[2:  ] + (0.5 + β[1:  ]) * U[1:-1]) \
        - mwc * ((0.5 - β[ :-1]) * U[1:-1] + (0.5 + β[ :-1]) * U[ :-2]) \
        + αf[1:-1] * (P[1:-1, np.newaxis] - P[:-2, np.newaxis]) * 1e5 * A \
        + αf[1:-1] * ρf[1:-1] * g * np.cos(θ) * A * (H[1:-1, np.newaxis] - H[:-2, np.newaxis])  \
        + αf[1:-1] * ρf[1:-1] * g * np.sin(θ) * ΔV \
        + τw[1:-1] * (Swf[1:-1] / A) * ΔV + sign_τ * τi[1:-1, np.newaxis] * (Sif[1:-1, np.newaxis] / A) * ΔV \
        + αf[1:-1] * (Pd[1:-1, np.newaxis] - Pd[:-2, np.newaxis]) * A
    # left side momentum
    f[-1,:nphases] += \
        + ρf[-1] *  U[-1] * dtαc[-1] * ΔV * 0.5 \
        + ρf[-1] * αf[-1] * dtU[-1] * ΔV * 0.5 \
        +  U[-1] * αf[-1] * c[-1] * dtPc[-1, np.newaxis] * 1e5 * ΔV * 0.5 \
        + α[-2] * ρ[-2] *  U[-1] * A * U[-1] \
        - α[-2] * ρ[-2] * Uc[-1] * A * ((0.5 - β[-1]) * U[-1] + (0.5 + β[-1]) * U[-2]) \
        + αf[-1] * (Ppresc - P[-2, np.newaxis]) * 1e5 * A \
        + αf[-1] * ρf[-1] * g * np.cos(θ) * A * (H[-1, np.newaxis] - H[-2, np.newaxis])  \
        + αf[-1] * ρf[-1] * g * np.sin(θ) * ΔV * 0.5 \
        + τw[-1] * (Swf[-1] / A) * ΔV * 0.5 + sign_τ * τi[-1, np.newaxis] * (Sif[-1, np.newaxis] / A) * ΔV * 0.5 \
        + αf[-1] * (Pd[-1, np.newaxis] - Pd[-2, np.newaxis]) * A
    
    # Momentum Eq. units are [kg/s * m/s], so we normalize it to [m3/m3]
    # by dividing by the speed of sound of the fluid and a ref density 
    # (ρref), and finally multiplying by dt.
    f[1:,:nphases] *= dt / (sound_speed * ρreff[1:] * ΔV)
    f[-1,:nphases] /= 0.5

    ######################################
    ######################################
    # MASS CENTRAL NODES        
    fp =  \
        + ρ[:-1] * dtα[:-1] * ΔV \
        + α[:-1] * c[:-1] * dtP[:-1, np.newaxis] * 1e5 * ΔV \
        + me - mw        
    # Mass Eq. units are [kg/s], so we normalize it to [m3/m3]
    # by dividing by the ref density (ρref) and multiplying by dt.    
    f[:-1, nphases:-1] += fp * dt / (ρref[:-1] * ΔV)
    ######################################
    
    # Boundary Momentum [kg/s], so we normalize it to [m3/m3]
    # by dividing by the ref density (ρref) and multiplying by dt.            
    f[0,:nphases] = -(Mpresc - αα[0] * ρρ[0] * U[0] * A) * dt / (ρref[0] * 0.5 * ΔV)
    
    # Mass BC (opening) (already dimensionless) [m3/m3]
    f[-1,nphases:-1] = α[-1] - α[-2]
    ######################################
    ######################################
    # END PHASE "LOOP"
    ######################################
    ######################################
    
    # Equation for pressure (already dimensionless) [m3/m3]
    f[:-1, -1] = 1 - αTotal[:-1]

    # pressure ghost    
    f[ -1, -1] = -(Ppresc - P[-1]) / Ppresc
    
    return f
