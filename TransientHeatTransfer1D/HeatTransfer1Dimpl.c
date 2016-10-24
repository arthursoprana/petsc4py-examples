#include "HeatTransfer1Dimpl.h"


#undef __FUNCT__
#define __FUNCT__ "FormInitGuess"
/* 
   FormInitialGuess - Forms initial approximation.

   Input Parameters:
   X - vector
   p - user parameters

   Output Parameter:
   X - vector
 */
PetscErrorCode FormInitGuess(DM da,Vec X,Params *p)
{
  PetscInt       i, xs, xm;
  PetscErrorCode ierr;
  Field          *x;

  /*
     Get local grid boundaries (for 1-dimensional DMDA):
       xs   - starting grid indices (no ghost points)
       xm   - widths of local grid (no ghost points)
  */
  ierr = DMDAGetCorners(da,&xs,PETSC_NULL,PETSC_NULL,&xm,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

  /*
     Get a pointer to vector data.
       - For default PETSc vectors, VecGetArray() returns a pointer to
         the data array.  Otherwise, the routine is implementation dependent.
       - You MUST call VecRestoreArray() when you no longer need access to
         the array.
  */
  ierr = DMDAVecGetArray(da,X,&x);CHKERRQ(ierr);

  /*
     Compute initial guess over the locally owned part of the grid
     Initial condition is motionless fluid and equilibrium temperature
  */

    for (i=xs; i<xs+xm; i++) {
      x[i].T = p->temperature_left_;
    }

  /*
     Restore vector
  */
  ierr = DMDAVecRestoreArray(da,X,&x);CHKERRQ(ierr);
  return 0;
}



#undef __FUNCT__
#define __FUNCT__ "FormFunctionLocal"
PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, PetscReal t, Field *x, Field *x_t, Field *f, Params *p)
 {
    PetscErrorCode ierr;
    PetscInt       xints,xinte,i;
    PetscReal      dx;
    PetscReal      L, k, conductivity, T_left, T_right, Q;

    PetscFunctionBegin;
    conductivity = p->conductivity_;  
    T_left = p->temperature_left_;
    T_right  = p->temperature_right_;
    Q = p->source_term_;
    L = p->wall_length_;
    /* 
     Define mesh intervals ratios for uniform grid.
    */

    dx = L / (PetscReal)(info->mx-1);
    k = conductivity;

    xints = info->xs; xinte = info->xs+info->xm; 
  
    /* Test whether we are on the left edge of the global array */
    if (xints == 0) {
        i = 0;
        xints = xints + 1;
        /* left edge */
        f[i].T  = x[i].T - T_left;
    }

    /* Test whether we are on the right edge of the global array */
    if (xinte == info->mx) {
        i = info->mx - 1;
        xinte = xinte - 1;
        /* right edge */ 
        f[i].T  = x[i].T - T_right;
    }

    /* Compute over the interior points */
    for (i=xints; i<xinte; i++) {
        /* Temperature */
        f[i].T = x_t[i].T * dx
               -1.0 * (+ k * (x[i + 1].T - x[i].T) / dx
                       - k * (x[i].T - x[i - 1].T) / dx + Q * dx);
    }
  /*
     Flop count (multiply-adds are counted as 2 operations)
  */
  ierr = PetscLogFlops(84.0*info->xm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 


#undef __FUNCT__
#define __FUNCT__ "FormFunction"
PetscErrorCode FormFunction(TS ts, PetscReal t, Vec X, Vec X_t, Vec F, Params *p)
{
    DMDALocalInfo  info;
    Field          *u, *u_t, *fu;
    PetscErrorCode ierr;
    Vec            localX, localX_t;
    DM             da;
    PetscFunctionBegin;

    ierr = TSGetDM(ts, &da); CHKERRQ(ierr);

    ierr = DMGetLocalVector(da, &localX); CHKERRQ(ierr);
    ierr = DMGetLocalVector(da, &localX_t); CHKERRQ(ierr);
    /*
    Scatter ghost points to local vector, using the 2-step process
    DMGlobalToLocalBegin(), DMGlobalToLocalEnd().
    */
    ierr = DMGlobalToLocalBegin(da, X, INSERT_VALUES, localX); CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da, X, INSERT_VALUES, localX); CHKERRQ(ierr);

    ierr = DMGlobalToLocalBegin(da, X_t, INSERT_VALUES, localX_t); CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da, X_t, INSERT_VALUES, localX_t); CHKERRQ(ierr);

    ierr = DMDAGetLocalInfo(da, &info); CHKERRQ(ierr);

    ierr = DMDAVecGetArray(da, localX, &u); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, localX_t, &u_t); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, F, &fu); CHKERRQ(ierr);

    ierr = FormFunctionLocal(&info, t, u, u_t, fu, p); CHKERRQ(ierr);

    ierr = DMDAVecRestoreArray(da, localX, &u); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, localX_t, &u_t); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, F, &fu); CHKERRQ(ierr);

    ierr = DMRestoreLocalVector(da, &localX); CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(da, &localX_t); CHKERRQ(ierr);

    PetscFunctionReturn(0);
}
