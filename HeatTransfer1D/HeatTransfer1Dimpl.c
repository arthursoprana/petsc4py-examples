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
  PetscInt       i, j, mx, xs, xm;
  PetscErrorCode ierr;
  PetscReal      conductivity, dx;
  Field          *x;

  
  printf("temperature left: %e\n", p->temperature_left_);
  printf("temperature right: %e\n", p->temperature_right_);
  printf("conductivity: %e\n", p->conductivity_);

  conductivity = p->conductivity_;

  ierr = DMDAGetInfo(da,0,&mx,0,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  dx  = 1.0/(mx-1);

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
PetscErrorCode FormFunctionLocal(DMDALocalInfo *info,Field *x,Field *f,Params *p)
 {
    PetscErrorCode ierr;
    PetscInt       xints,xinte,i;
    PetscReal      dx;
    PetscReal      L, k, conductivity, T_left, T_right;

    PetscFunctionBegin;
    conductivity = p->conductivity_;  
    T_left = p->temperature_left_;
    T_right  = p->temperature_right_;

    /* 
     Define mesh intervals ratios for uniform grid.

     Note: FD formulae below are normalized by multiplying through by
     local volume element (i.e. hx*hy) to obtain coefficients O(1) in two dimensions.

     
    */
    L = 1.0;
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
        f[i].T = k * (x[i+1].T -   x[i].T) / dx
               - k * (  x[i].T - x[i-1].T) / dx;
    }
  /*
     Flop count (multiply-adds are counted as 2 operations)
  */
  /*ierr = PetscLogFlops(84.0*info->ym*info->xm);CHKERRQ(ierr);*/
  PetscFunctionReturn(0);
} 

#undef __FUNCT__
#define __FUNCT__ "FormFunction"
PetscErrorCode FormFunction(DM da, Vec X, Vec F, Params *p)
{
  DMDALocalInfo  info;
  Field          *u,*fu;
  PetscErrorCode ierr;
  Vec            localX;

  PetscFunctionBegin;
  ierr = DMGetLocalVector(da,&localX);CHKERRQ(ierr);
  /*
     Scatter ghost points to local vector, using the 2-step process
        DMGlobalToLocalBegin(), DMGlobalToLocalEnd().
  */
  ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,localX,&u);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,F,&fu);CHKERRQ(ierr);
  ierr = FormFunctionLocal(&info,u,fu,p);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,localX,&u);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,F,&fu);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localX);CHKERRQ(ierr);
  PetscFunctionReturn(0); 
}


