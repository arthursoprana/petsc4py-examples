import sys
import numpy as np
import petsc4py
from petsc4py import PETSc
from matplotlib import pyplot as plt
import CompositeSimple1D



petsc4py.init(sys.argv)

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
    fi_bubble = 1e0
    
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
    f = correct_friction_factor(D, H, f, hgc=0.001)
    
    return np.minimum(f, 100)


def ideal_gas_density_model(P, deriv=False):
    Z = 1.0
    R = 8.315
    T = 300.0
    M = 16.0e-3
        
    if deriv:
        return 0 * P + 1 / (Z * (R/M) * T) # 1/a**2
    else:
        return P * M / (Z * R * T)
    

def constant_density_model(P, deriv=False):
        
    if deriv:
        return 0 * P
    else:
        return 1000 + 0*P
    
def liquid_viscosity_model(P):
    return 1e-3 + P*0

def gas_viscosity_model(P):
    return 1e-5 + P*0
        
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
    
def calculate_residual(UT, dtUT, αT, dtαT, P, dtP, dx, nx, dof, ρrefT, D, DhT, SwT, Si, H, fi=None):
    f = np.zeros((nx, dof))    
    
    nphases = αT.shape[1] 
    
    Ppresc  = 1.0 # [bar]
    USpresc = [3.0, 0.6] # [m/s]
    
                      
    A = 0.25 * np.pi * D ** 2 # [m]
    ΔV = A * dx
    
    ρg = density_model[0](P*1e5)
    
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

        f[1:, phase] /= USpresc[phase] * ρref[:-1]
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
        
        # boundaries            
        # Momentum            
        f[ 0,phase] = USpresc[phase] - U[0] * αf[0]
        # Mass
        f[-1,phase+nphases] = α[-2] - α[-1]
    
    f[:-1, -1] += 1  #  αG + αL = 1
    
    # pressure ghost    
    f[ -1, -1] = Ppresc - 0.5 * (P[-1] + P[-2])
    
    return f
    

def computeGeometricProperties(α, D):
    assert α.shape[1] == 2, 'Only 2 phases supported!'
    
    δ  = ComputeSectorAngle(α)        
        
    angle = δ[:,1]
    Si = D * np.sin(0.5 * angle)
    Sw = δ * D  
    H = 0.5 * D * (1.0 - np.cos(0.5 * angle))

    A = 0.25 * np.pi * D ** 2 
    Dh = np.zeros_like(α)
    Dh[:, 0]  = 4.0 * α[:, 0] * A / (Sw[:, 0] + Si) # Closed channel for gas
    Dh[:, 1]  = 4.0 * α[:, 1] * A / (Sw[:, 1]) # Open channel for liquid
    
    return Dh, Sw, Si, H


class Flow(object):
    def __init__(self, dm, nx, dof, pipe_length, nphases, α0):

        self.dm  = dm        
        self.L   = pipe_length
        self.nx  = nx
        self.dof = dof
        self.nphases = nphases
        self.D =  0.1 # [m]   
        self.Pold = None
        self.α = α0
        self.ρref = np.zeros((nx, nphases))
        for phase in range(nphases):
            self.ρref[:, phase] = density_model[phase](1e5)
            
        self.Dh = None
        self.Sw = None
        self.Si = None
        self.H  = None
        self.fi = np.zeros(nx)
        self.Dh, self.Sw, self.Si, self.H = computeGeometricProperties(α0, self.D)
    
    def updateFunction(self, snes, step):
        # Compute geometric properties
        self.Dh, self.Sw, self.Si, self.H = computeGeometricProperties(self.α, self.D)
        
        nphases = self.nphases
        
        # Compute ref density
        for phase in range(nphases):
            self.ρref[:, phase] = density_model[phase](self.Pold*1e5)
            
        sol = snes.getSolution()[...]
        u = sol.reshape(nx, dof)
        
        U = u[:, 0:nphases]
        α = u[:, nphases:-1]
        P = u[:, -1]

        u[:, nphases:-1] = np.where(u[:, nphases:-1] < 0.0, 0.0, u[:, nphases:-1])
        u[:, nphases:-1] = np.where(u[:, nphases:-1] > 1.0, 1.0, u[:, nphases:-1])
        
#         snes.getSolution()[...] = u.flatten()
        
        ρg = density_model[0](P*1e5)
        μg = viscosity_model[0](P*1e5)
        Dhg = self.Dh[:, 0]
    
        Ur = np.abs(U[:, 0] - U[:, 1])
        Rei = ρg * np.abs(Ur) * Dhg / μg + 1e-3
        D = self.D
        A = 0.25 * np.pi * D ** 2 
        fi = andreussi_gas_liquid(
            Rei,
            α[:, 0],
            D,
            1e-5,
            self.H,
            density_model[1](P*1e5),
            ρg,
            Ur,
            A * α[:, 0]
        )

        self.fi = fi

        
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
                        
                        self.α = α
                        self.Pold = P - dt * dtP   
                        dx = L / (nx - 1)
                        
                        residual = calculate_residual(U, dtU, α, dtα, P, dtP, dx, nx, dof, self.ρref, 
                                                      self.D, self.Dh, self.Sw, self.Si, self.H, None)
                        F.setArray(residual.flatten())


def transient_heat_transfer_1D(
    npipes, nx, dof, nphases,
    pipe_length,
    initial_time,
    final_time,
    initial_time_step,
    initial_solution,    
    impl_python=False
    ):
    
    # Time Stepper (TS) for ODE and DAE
    # DAE - https://en.wikipedia.org/wiki/Differential_algebraic_equation
    # https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/TS/
    ts = PETSc.TS().create()

    # http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/DM/index.html
    pipes = []
    for i in range(npipes):
        boundary_type = PETSc.DMDA.BoundaryType.GHOSTED
        da = PETSc.DMDA().create([nx], dof=dof, stencil_width=1, stencil_type='star', boundary_type=boundary_type)
        da.setFromOptions()
        pipes.append(da)
    
    # Create a redundant DM, there is no petsc4py interface (yet)
    # so we created our own wrapper
    # http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/DM/DMREDUNDANT.html
#     dmredundant = PETSc.DM().create()
#     dmredundant.setType(dmredundant.Type.REDUNDANT)
#     CompositeSimple1D.redundantSetSize(dmredundant, 0, dof)
#     dmredundant.setDimension(1)
#     dmredundant.setUp()

    # http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/DM/DMCOMPOSITE.html
    dm = PETSc.DMComposite().create()
    
    for pipe in pipes:        
        dm.addDM(pipe)

#     dm.addDM(dmredundant)
#     CompositeSimple1D.compositeSetCoupling(dm)
    
    ts.setDM(dm)
    
    snes = ts.getSNES()

    if PETSc.Options().getString('snes_type') == snes.Type.VINEWTONRSLS:
        xl = np.zeros((nx, dof))
        xu = np.zeros((nx, dof))    
        xl[:] = -1e20
        xu[:] =  1e20   
        xl[:, nphases:-1] = 0.0
        xu[:, nphases:-1] = 1.0       
        xlVec = dm.createGlobalVec()
        xuVec = dm.createGlobalVec()    
        xlVec.setArray(xl.flatten())
        xuVec.setArray(xu.flatten())    
        snes.setVariableBounds(xlVec, xuVec)
    
    F = dm.createGlobalVec()

    if impl_python:     
        α0 = initial_solution.reshape((nx,dof))[:, nphases:-1]
        pde = Flow(dm, nx, dof, pipe_length, nphases, α0)
        ts.setIFunction(pde.evalFunction, F)
    else:
        # http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/TS/TSSetIFunction.html
        assert False, 'C function not implemented yet!'
#         ts.setIFunction(CompositeSimple1D.formFunction, F,
#                          args=(conductivity, source_term, wall_length, temperature_presc))    
    
    snes = ts.getSNES()
    
    snes.setUpdate(pde.updateFunction)
    
    x = dm.createGlobalVec()    

    x[...] = initial_solution

    # http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/TS/TSSetDuration.html
    ts.setDuration(max_time=final_time, max_steps=None)
    
    # http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/TS/TSSetInitialTimeStep.html
    ts.setInitialTimeStep(initial_time=initial_time, initial_time_step=initial_time_step)
    
    # http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/TS/TSSetProblemType.html
    ts.setProblemType(ts.ProblemType.NONLINEAR)
    
    
    ts.setFromOptions()

    ts.solve(x)

    return x

options = PETSc.Options()
options.clear()

dt = 0.001           # [s]
dt_min = 0.001         # [s]
dt_max = 10.0               # [s]

# time_intervals = [1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 0.2, 10]
# time_intervals = [1e-4, 1e-3]
time_intervals = np.linspace(0,250, num=2500)
# time_intervals = np.concatenate((np.logspace(-4, -0.1, num=200), np.linspace(1,250))) 


# TS config
options.setValue('-ts_type', 'bdf') # https://en.wikipedia.org/wiki/Backward_differentiation_formula
options.setValue('-ts_bdf_order', 1) # https://en.wikipedia.org/wiki/Backward_differentiation_formula
options.setValue('-ts_adapt_type', 'basic') # basic or none
options.setValue('-ts_bdf_adapt', None)
options.setValue('-ts_adapt_dt_min', dt_min)
options.setValue('-ts_adapt_dt_max', dt_max)
options.setValue('-ts_exact_final_time', 'matchstep') # http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/TS/TSSetExactFinalTime.html
options.setValue('-ts_adapt_scale_solve_failed', 0.5) # Scale step by this factor if solve fails ()

options.setValue('-ts_adapt_basic_clip', (0.1, 1.1)) # Admissible decrease/increase factor in step size (TSAdaptBasicSetClip)
options.setValue('-ts_adapt_basic_safety', 1.0) # Safety factor relative to target error ()
options.setValue('-ts_adapt_basic_reject_safety', 0.5) # Extra safety factor to apply if the last step was rejected ()
options.setValue('-ts_adapt_basic_always_accept', False) # Always accept the step regardless of whether local truncation error meets goal ()
options.setValue('-ts_monitor', None)


dense = False
if dense:
    options.setValue('-dmcomposite_dense_jacobian', None)
else:
    options.delValue('-dmcomposite_dense_jacobian')

options.setValue('-ts_fd_color', None)

options.setValue('-mat_fd_type', 'ds')
# options.setValue('-mat_fd_coloring_err', 1e-5)
# options.setValue('-mat_fd_coloring_umin', 1e-12)


options.setValue('-ts_max_snes_failures', -1)

# options.setValue('-mat_view', 'draw')
# options.setValue('-draw_pause', 1000)
#options.setValue('-is_coloring_view', '')
options.setValue('-help', None)

options.setValue('-snes_monitor_short', None)
options.setValue('-snes_converged_reason', None)

# options.setValue('-sub_0_snes_monitor_short', None)
# options.setValue('-sub_0_snes_converged_reason', None)
# options.setValue('-sub_1_snes_monitor_short', None)
# options.setValue('-sub_1_snes_converged_reason', None)

# options.setValue('-sub_0_fieldsplit_0_ksp_monitor_short', None)
# options.setValue('-sub_0_fieldsplit_1_ksp_monitor_short', None)
# options.setValue('-sub_1_fieldsplit_0_ksp_monitor_short', None)
# options.setValue('-sub_1_fieldsplit_1_ksp_monitor_short', None)
# options.setValue('-sub_0_fieldsplit_0_ksp_monitor_short', None)
# options.setValue('-sub_0_fieldsplit_1_ksp_monitor_short', None)
# options.setValue('-sub_1_fieldsplit_0_ksp_monitor_short', None)
# options.setValue('-sub_1_fieldsplit_1_ksp_monitor_short', None)
# 
# options.setValue('-sub_0_fieldsplit_0_ksp_converged_reason', None)
# options.setValue('-sub_0_fieldsplit_1_ksp_converged_reason', None)
# options.setValue('-sub_1_fieldsplit_0_ksp_converged_reason', None)
# options.setValue('-sub_1_fieldsplit_1_ksp_converged_reason', None)
# options.setValue('-sub_0_fieldsplit_0_ksp_converged_reason', None)
# options.setValue('-sub_0_fieldsplit_1_ksp_converged_reason', None)
# options.setValue('-sub_1_fieldsplit_0_ksp_converged_reason', None)
# options.setValue('-sub_1_fieldsplit_1_ksp_converged_reason', None)

# options.setValue('-snes_convergence_test', 'skip')
# options.setValue('-snes_max_it', 10)
# options.setValue('-snes_linesearch_type', 'basic')
# options.setValue('-snes_function_type', 'preconditioned')
# options.setValue('-snes_type', 'nasm')
# 
# options.setValue('snes_nasm_type', 'basic') # <BASIC> (choose one of) NONE RESTRICT INTERPOLATE BASIC ()
# options.setValue('snes_nasm_damping', 1 ) #The new solution is obtained as old solution plus dmp times (sum of the solutions on the subdomains) (SNESNASMSetDamping)
# options.setValue('snes_nasm_sub_view', False) # <FALSE> Print detailed information for every processor when using -snes_view ()
# options.setValue('snes_nasm_finaljacobian', False) #: <FALSE> Compute the global jacobian of the final iterate (for ASPIN) ()
# options.setValue('snes_nasm_finaljacobian_type', 'finalouter') #<FINALOUTER> (choose one of) FINALOUTER FINALINNER INITIAL ()
# options.setValue('snes_nasm_log', True)   #: <TRUE> Log times for subSNES solves and restriction ()

# options.setValue('-snes_type', 'aspin')
# options.setValue('-snes_npc_side', 'left')
# options.setValue('-npc_snes_type', 'nasm')
# options.setValue('-npc_snes_nasm_type', 'restrict')

options.setValue('-snes_vi_zero_tolerance', 1e-8) # Tolerance for considering x[] value to be on a bound (None)
options.setValue('-snes_vi_monitor', True) # Monitor all non-active variables (SNESMonitorResidual)
# options.setValue('-snes_vi_monitor_residual', True) # Monitor residual all non-active variables; using zero for active constraints (SNESMonitorVIResidual)
   
options.setValue('-snes_type', 'composite')
options.setValue('-snes_composite_type', 'additiveoptimal')

# newtonls newtontr test nrichardson ksponly vinewtonrsls 
# vinewtonssls ngmres qn shell ngs ncg fas ms nasm anderson 
# aspin composite python (SNESSetType)
# options.setValue('-snes_composite_sneses', 'ksponly,newtonls,nrichardson')
# options.setValue('-snes_composite_sneses', 'vinewtonrsls,newtonls')
# options.setValue('-snes_composite_sneses', 'ksponly,vinewtonrsls')
options.setValue('-snes_composite_sneses', 'fas,newtonls')
# options.setValue('-snes_composite_sneses', 'ksponly,newtonls')
# options.setValue('-snes_composite_damping', 0.5) # Damping of the additive composite solvers (SNESCompositeSetDamping)
options.setValue('-snes_composite_stol', 0.1) # Step tolerance for restart on the additive composite solvers ()
options.setValue('-snes_composite_stol', 1.1) # Residual tolerance for the additive composite solvers ()


# options.setValue('-snes_composite_sneses', 'fas,newtonls')
# options.setValue('-snes_composite_sneses', 'ngmres,newtonls')
 
# options.setValue('da_refine_x', 1)
# options.setValue('sub_0_fas_levels_snes_type', 'gs')
# options.setValue('sub_0_fas_levels_snes_max_it', 6)
# options.setValue('sub_0_fas_coarse_snes_type', 'ngs')
# options.setValue('sub_0_fas_coarse_snes_max_it', 6)
# options.setValue('sub_0_fas_coarse_snes_linesearch_type', 'basic')
# options.setValue('sub_1_snes_linesearch_type', 'basic')
# options.setValue('sub_1_pc_type', 'mg')

# options.setValue('sub_0_snes_convergence_test', 'skip')
# options.setValue('sub_0_snes_max_it', 10)




# # for field split solver
# # for snes in ['sub_0_', 'sub_1_']: # for snes == composite 
# # for snes in ['sub_0_']: # for snes == composite 
# for snes in ['sub_']: # for snes == composite 
# # for snes in ['']:
# # for snes in ['', 'npc_']:
# # for snes in ['npc_']:
#     # ksp config
#     options.setValue(snes + 'ksp_type', 'fgmres')  
#              
#     # pc config
#     # For direct solver
#     # options.setValue(snes + 'pc_type', 'lu')
#     # options.setValue(snes + 'ksp_type', 'preonly')
# #     options.setValue(snes + 'pc_factor_shift_type', 'NONZERO')
# #     options.setValue(snes + 'pc_factor_shift_amount', 1e-12)
#                  
#     options.setValue(snes + 'pc_type', 'fieldsplit')  
#     options.setValue(snes + 'pc_fieldsplit_type', 'schur')  
#     options.setValue(snes + 'pc_fieldsplit_schur_fact_type', 'lower')   
#     options.setValue(snes + 'pc_fieldsplit_block_size', 5)   
#     options.setValue(snes + 'pc_fieldsplit_0_fields', '0,1')   
#     options.setValue(snes + 'pc_fieldsplit_1_fields', '2,3,4')   
#                  
# #     options.setValue(snes + 'fieldsplit_0_ksp_rtol', 1000)
# #     options.setValue(snes + 'fieldsplit_0_ksp_atol', 1000)
# #     options.setValue(snes + 'fieldsplit_0_ksp_convergence_test', 'skip')
# #     options.setValue(snes + 'fieldsplit_1_ksp_convergence_test', 'skip')
#      
#     options.setValue(snes + 'fieldsplit_0_ksp_type', 'gmres')
#     options.setValue(snes + 'fieldsplit_0_pc_type', 'bjacobi')
#     options.setValue(snes + 'fieldsplit_0_sub_pc_type', 'bjacobi')                 
#                  
#     options.setValue(snes + 'fieldsplit_1_pc_type', 'jacobi') 
#     options.setValue(snes + 'fieldsplit_1_pc_jacobi_type', 'diagonal') # (choose one of) DIAGONAL ROWMAX ROWSUM (PCJacobiSetType)
#     options.setValue(snes + 'fieldsplit_1_ksp_type', 'preonly') 
#                  
#     options.setValue(snes + 'fieldsplit_1_inner_pc_type', 'jacobi')
#     options.setValue(snes + 'fieldsplit_1_inner_ksp_type', 'preonly')
#                  
#     options.setValue(snes + 'fieldsplit_1_upper_ksp_type', 'preonly')
#     options.setValue(snes + 'fieldsplit_1_upper_pc_type', 'jacobi')
#     options.setValue(snes + 'fieldsplit_1_upper_pc_jacobi_type', 'diagonal') # (choose one of) DIAGONAL ROWMAX ROWSUM (PCJacobiSetType)
#                  
#     options.setValue(snes + 'fieldsplit_1_mat_schur_complement_ainv_type', 'lump')   
#     options.setValue(snes + 'pc_fieldsplit_schur_precondition', 'selfp')   
# #     options.setValue(snes + 'pc_fieldsplit_schur_precondition', 'a11')   
# #     options.setValue(snes + 'pc_fieldsplit_detect_saddle_point', None)
#     # options.setValue(snes + 'pc_fieldsplit_default', None)
     

npipes = 1
nx = 1000

nphases = 2
dof = nphases * 2 + 1

pipe_length = 1000.0 # [m]

f, axarr = plt.subplots(4, sharex=True)
axarr[0].set_title('Results')

initial_solution = np.zeros((nx,dof))
initial_solution[:,0:nphases] = 0.0001 # Velocity

αG = 0.01
initial_solution[:,2] = αG # vol frac
initial_solution[:,3] = 1-αG # vol frac

initial_solution[:,-1] = 1.0  # Pressure
initial_time = 0.0

sols = []
for final_time in time_intervals:

    sol = transient_heat_transfer_1D(
        npipes, nx, dof, nphases,
        pipe_length,
        initial_time,
        final_time,
        dt,
        initial_solution.flatten(),
        impl_python=True
        )
    
    SOL = sol[...].reshape(nx, dof)

    U = SOL[:, 0:nphases]
    α = SOL[:, nphases:-1]
    P = SOL[:, -1]
    
    initial_solution[:,0:nphases] = U # Velocity
    initial_solution[:,nphases:-1] = α # vol frac
    initial_solution[:,-1] = P   # Pressure

    initial_time = final_time

                        
    sols.append((U, P))
    
    dx = pipe_length / (nx - 1)
    x = np.linspace(0, npipes*pipe_length, npipes*nx, endpoint=True) + 0.5*dx
    xx = np.concatenate((x[:-1], [pipe_length]))


    UU = U
    PP = np.concatenate((P[:-1], [1.0]))
    αα = α
    
    axarr[0].cla()
    axarr[1].cla()
    axarr[2].cla()
    axarr[3].cla()
    axarr[0].plot(xx, αα, '.-')
    axarr[1].plot(xx, UU, '.-')
    axarr[2].plot(xx, αα*UU, '.-')
    axarr[3].plot(xx, PP, 'r.-')
    plt.xlim(0, pipe_length)
    axarr[0].set_ylim(0, 1)
#     axarr[1].set_ylim(0, 4)
#     axarr[3].set_ylim(0.99, 1.01)
#     plt.show()
    
    plt.draw()
    plt.pause(0.0001)

