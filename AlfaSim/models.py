import numpy as np

GRAVITY_CONSTANT = 9.81 # [m/s2]

def colebrook_white_explicit_friction_factor(reynolds, volume_fraction, diameter, absolute_rugosity):
    '''
    From Biberg 2008 - APPENDIX E: AN EXPLICIT APPROXIMATION FOR THE
                       COLEBROOK-WHITE FORMULA
    '''
    Re = np.abs(reynolds + 1.0e-15)
    ks = absolute_rugosity
    D  = diameter
    
    invf_haland = -1.8 * np.log10(6.9 / Re + (ks / (3.7 * D)) ** 1.11)
    invf_haland = np.where(invf_haland < 0.0, 0.0, invf_haland)
    
    invf0 = invf_haland
    
    # There is a 1/lambda_s (1/f_s) in the formula, but Biberg did not define this 
    # variable, so we'll use 1/f_0 (1/lambda_0)
    invfs = invf0
    
    term = (2.51 * invf0 / Re + ks / (3.7 * D))
    
    invf = ((5.02 * invfs / Re) - 4.6 * term * np.log10(term)) / ((5.02 / Re) + 2.3 * term)
    f_w_turbulent = 1.0 / invf ** 2
    
    # Avoid numpy warnings
    f_w_turbulent[np.where(np.isnan(f_w_turbulent))] = 0.0
    
    f_w_laminar = 64.0 / Re
    
    f_w = np.where(Re < 100.0, f_w_laminar, np.maximum(f_w_laminar, f_w_turbulent))
    
    fanning_factor = 0.25
    return fanning_factor * f_w


def gas_wall_taitel_dukler(reynolds, volume_fraction, diameter, absolute_rugosity):
    '''
    Taitel and Dukler (1976), listed in Bonizzi 2009.
    
    '''
    Re = np.abs(reynolds + 1.0e-15)    
    return np.where(Re > 2100.0, 0.046 * (Re ** -0.2), 16.0 / Re) # Taitel and Dukler (1976), listed in Bonizzi 2009


def correct_friction_factor(diameter, liquid_height_center, interface_friction_factor, hgc=0.01):
    '''
    ###############################################################
    # GAS INTERFACIAL FRICTION FACTOR CORRECTION
    ###############################################################
    Based on various parametric studies, it was found that the most 
    important closure model is the interfacial friction at small gas 
    fractions. Indeed, if the usual interfacial friction factor in LedaFlow 
    is used when the gas fraction is small, the prevailing slug bubble 
    velocities become too high, suggesting that the default interfacial
    friction model is not applicable to this situation. The physical 
    argument for this formulation is that at small gas fractions, the 
    flow can no longer be modelled as separated; rather it is closer 
    to a kind of bubbly flow with relatively little gas-liquid slip.
    
    Calculate liquid height
    Define a critical height, above which fi will be modified
    Define a fi_bubble (emulate slip)
    '''
    fi = interface_friction_factor
    hg = diameter - liquid_height_center
    beta = hg / hgc
    
    # Tricky one!    
    #fi_bubble = drag_bubble_tomiyama(diameter_b, reynolds_b, liquid_density, gas_density, gas_liquid_surface_tension)
    fi_bubble = 1000
    
    G = 1.0 / (1.0 + np.exp(-100.0 * (beta - 1.0)))

    fi_new = fi * G + fi_bubble * (1.0 - G)

    return fi_new

def andreussi_gas_liquid(
        interfacial_reynolds,
        gas_volume_fraction,
        diameter,
        absolute_rugosity,
        liquid_height,
        liquid_density,
        gas_density,
        interfacial_vel,
        gas_area
    ): 
    f_g = gas_wall_taitel_dukler(interfacial_reynolds, gas_volume_fraction, diameter, absolute_rugosity)

    H = liquid_height
    D = diameter
    dAdHl = 2.0 * np.sqrt(H * (D - H))   


    F = np.where(np.abs(liquid_density - gas_density) < 1.0e-8,
                 0.0,
                 interfacial_vel * np.sqrt(gas_density / (liquid_density - gas_density) * (dAdHl / gas_area) * (1.0 / GRAVITY_CONSTANT)))
  
    F = np.where(np.logical_or(F < 0.36, np.isnan(F)), 0.36, F)
 
    factor = np.where(F > 0.36, 1.0 + 29.7 * ((F - 0.36) ** 0.67) * ((liquid_height / diameter) ** 0.2), 1.0)

    f = factor * f_g
    f = correct_friction_factor(D, H, f, hgc=0.01 * D)

    return f


def ideal_gas_density_model(P, deriv=False):
    a = 316.0
        
    if deriv:
        return 0 * P + 1 / a**2
    else:
        return P / a**2
    

def constant_density_model(P, deriv=False):
        
    if deriv:
        return 0 * P
    else:
        return 1000 + 0*P
    
def liquid_viscosity_model(P):
    return 5e-2 + P*0

def gas_viscosity_model(P):
    return 5e-6 + P*0
        
density_model = [ideal_gas_density_model, constant_density_model]
viscosity_model = [gas_viscosity_model, liquid_viscosity_model]

def ComputeSectorAngle(volume_fraction, extra_precision=True):
    '''
    Reference: \Google Drive\ALFA Sim\books\PipeFlow2 - Multi-phase Flow Assurance (Bratland).pdf
    Eq. 3.4.5
    OBS: The equation was generalized to compute the angle for each layer, then the expression is a bit different
    
    :param bool extra_precision:
        From Biberg 2008 - APPENDIX D, eq. 130, extra precision so max error is
        ~ 0.00005 rad.
    '''        
    
    # Keep values inside 0,1 range for the calculations below
    volume_fraction_fixed = np.where(volume_fraction > 1.0, 1.0, volume_fraction)
    volume_fraction_fixed = np.where(volume_fraction_fixed < 0.0, 0.0, volume_fraction_fixed)
    
    term_1 = 1. - 2. * volume_fraction_fixed + volume_fraction_fixed ** (1. / 3.) - (1. - volume_fraction_fixed) ** (1. / 3.)
    beta = np.pi * volume_fraction_fixed + (3. * np.pi / 2.) ** (1. / 3.) * term_1
    
    if extra_precision:
        beta -= (0.005 * volume_fraction_fixed)     \
              * (1.0 - volume_fraction_fixed)       \
              * (1.0 - 2.0 * volume_fraction_fixed) \
              * (1.0 + 4.0 * (volume_fraction_fixed ** 2.0 + (1.0 - volume_fraction_fixed) ** 2.0))
                       
    return 2. * beta

def computeGeometricProperties(α, D):
    assert α.shape[1] == 2, 'Only 2 phases supported!'
    
    δ  = ComputeSectorAngle(α)        
        
    angle = δ[:,1]
    Si = D * np.sin(0.5 * angle)
    Sw = 0.5 * δ * D  
    H = 0.5 * D * (1.0 - np.cos(0.5 * angle))
    
    H = np.minimum(H, D)
    H = np.maximum(H, 0)
    
    A = 0.25 * np.pi * D ** 2 
    Dh = np.zeros_like(α)
    Dh[:, 0]  = 4.0 * α[:, 0] * A / (Sw[:, 0] + Si) # Closed channel for gas
    Dh[:, 1]  = 4.0 * α[:, 1] * A / (Sw[:, 1]) # Open channel for liquid
    
    return Dh, Sw, Si, H