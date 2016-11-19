#include "HeatTransfer1Dimpl.h"
#include "include_patch_pack.h"

#undef __FUNCT__
#define __FUNCT__ "FormInitGuess"

#define NETWORK_SIZE 2
/*
FormInitialGuessComp -
Forms the initial guess for the composite model
Unwraps the global solution vector and passes its local pieces into the user functions
*/
PetscErrorCode FormInitGuess(DM dm, Vec X, Params *p)
{
    PetscErrorCode ierr;    
    PetscInt nDM;
    // TODO: Find a way to dynamically allocate these arrays in c
    DM  das[NETWORK_SIZE];
    Vec Xs[NETWORK_SIZE];
   
    PetscFunctionBegin;

    ierr = DMCompositeGetNumberDM(dm, &nDM); CHKERRQ(ierr);
    ierr = DMCompositeGetEntriesArray(dm, das); CHKERRQ(ierr);

    /* Access the subvectors in X */
    ierr = DMCompositeGetAccessArray(dm, X, nDM, NULL, Xs); CHKERRQ(ierr);

    /* Evaluate local user provided function */
    for (int i = 0; i < nDM; ++i) {
        ierr = FormInitGuessLocal(das[i], Xs[i], p); CHKERRQ(ierr);
    }

    ierr = DMCompositeRestoreAccessArray(dm, X, nDM, NULL, Xs); CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormInitGuessLocal"
/* 
   FormInitialGuess - Forms initial approximation.

   Input Parameters:
   X - vector
   p - user parameters

   Output Parameter:
   X - vector
 */
PetscErrorCode FormInitGuessLocal(DM da, Vec X, Params *p)
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
  PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "FormFunctionLocal1"
PetscErrorCode FormFunctionLocal1(DMDALocalInfo *info, PetscReal t, Field *x1, Field *x2, Field *x_t, Field *f, Params *p, PetscInt comp_offset, PetscInt total_size)
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

    dx = L / (PetscReal)(total_size-1);
    k = conductivity;

    xints = info->xs; xinte = info->xs+info->xm;
  
    /* Test whether we are on the left edge of the global array */
    if (xints + comp_offset == 0) {
        i = 0;
        /* left edge */
        f[i].T  = x1[i].T - T_left;
    }

    // Not sure why we need this :(
    if (xints == 0) {
        xints = xints + 1;
    }
    if (xinte == info->mx) {
        xinte = xinte - 1;
        i = info->mx - 1;
        /* Temperature */
        f[i].T = x_t[i].T * dx
               - 1.0 * (+k * (x2[0].T - x1[i].T) / dx
                        -k * (x1[i].T - x1[i - 1].T) / dx + Q * dx);
    }

    /* Compute over the interior points */
    for (i=xints; i<xinte; i++) {
        /* Temperature */
        f[i].T = x_t[i].T * dx
               -1.0 * (+ k * (x1[i + 1].T - x1[i].T) / dx
                       - k * (x1[i].T - x1[i - 1].T) / dx + Q * dx);
    }
  /*
     Flop count (multiply-adds are counted as 2 operations)
  */
  ierr = PetscLogFlops(84.0*info->xm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 

#undef __FUNCT__
#define __FUNCT__ "FormFunctionLocal2"
PetscErrorCode FormFunctionLocal2(DMDALocalInfo *info, PetscReal t, Field *x1, Field *x2, Field *x_t, Field *f, Params *p, PetscInt comp_offset, PetscInt total_size)
{
    PetscErrorCode ierr;
    PetscInt       xints, xinte, i;
    PetscReal      dx;
    PetscReal      L, k, conductivity, T_left, T_right, Q;

    PetscFunctionBegin;
    conductivity = p->conductivity_;
    T_left = p->temperature_left_;
    T_right = p->temperature_right_;
    Q = p->source_term_;
    L = p->wall_length_;
    /*
    Define mesh intervals ratios for uniform grid.
    */

    dx = L / (PetscReal)(total_size - 1);
    k = conductivity;

    xints = info->xs; xinte = info->xs + info->xm;

    /* Test whether we are on the right edge of the global array */
    if (xinte + comp_offset == total_size) {
        i = info->mx - 1;
        /* right edge */
        f[i].T = x2[i].T - T_right;
    }

    // Not sure why we need this :(
    if (xints == 0) {
        xints = xints + 1;
        i = 0;
        /* Temperature */
        f[i].T = x_t[i].T * dx
            - 1.0 * (+k * (x2[i+1].T - x2[i].T) / dx
                     -k * (x2[i].T - x1[info->mx - 1].T) / dx + Q * dx);
    }
    if (xinte == info->mx) {
        xinte = xinte - 1;
    }

    /* Compute over the interior points */
    for (i = xints; i<xinte; i++) {
        /* Temperature */
        f[i].T = x_t[i].T * dx
               - 1.0 * (+k * (x2[i + 1].T - x2[i].T) / dx
                        -k * (x2[i].T - x2[i - 1].T) / dx + Q * dx);
    }
    /*
    Flop count (multiply-adds are counted as 2 operations)
    */
    ierr = PetscLogFlops(84.0*info->xm); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FormFunction"
PetscErrorCode FormFunction(TS ts, PetscReal t, Vec X, Vec X_t, Vec F, Params *p)
{
    DMDALocalInfo  info;
    Field          *u1, *u2;
    Field          *u1_t, *u2_t;
    Field          *f1, *f2;
    PetscErrorCode ierr;
    DM             dm;
    PetscInt nDM;

    // TODO: Find a way to dynamically allocate these arrays in c
    DM  das[NETWORK_SIZE];
    Vec Xs[NETWORK_SIZE], X_ts[NETWORK_SIZE], Fs[NETWORK_SIZE];
    Field* u[NETWORK_SIZE];
    Field* u_t[NETWORK_SIZE];
    Field* f[NETWORK_SIZE];

    PetscFunctionBegin;

    ierr = TSGetDM(ts, &dm); CHKERRQ(ierr);    
    ierr = DMCompositeGetNumberDM(dm, &nDM); CHKERRQ(ierr);
    ierr = DMCompositeGetEntriesArray(dm, das); CHKERRQ(ierr);
    
    /* Access the subvectors in X */    
    ierr = DMCompositeGetLocalVectorsArray(dm, nDM, NULL, Xs); CHKERRQ(ierr);
    ierr = DMCompositeScatterArray(dm, X, Xs); CHKERRQ(ierr);
    
    ierr = DMCompositeGetLocalVectorsArray(dm, nDM, NULL, X_ts); CHKERRQ(ierr);
    ierr = DMCompositeScatterArray(dm, X_t, X_ts); CHKERRQ(ierr);

    // Calculate total size
    PetscInt total_size = 0;
    for (int i = 0; i < nDM; ++i) {
        ierr = DMDAGetLocalInfo(das[i], &info); CHKERRQ(ierr);
        total_size += info.mx;
    }

    /* Access the subvectors in F.
    These are not ghosted so directly access the memory locations in F */
    ierr = DMCompositeGetAccessArray(dm, F, nDM, NULL, Fs); CHKERRQ(ierr);

    for (int i = 0; i < nDM; ++i) {
        ierr = DMDAVecGetArray(das[i], X_ts[i], &(u_t[i])); CHKERRQ(ierr);
        ierr = DMDAVecGetArray(das[i], Xs[i], &(u[i])); CHKERRQ(ierr);
        ierr = DMDAVecGetArray(das[i], Fs[i], &(f[i])); CHKERRQ(ierr);
    }

    PetscInt comp_offset = 0;

    // Obs for fluxes we need all u1 and u2 !
    ///////////////////////////////////////////////////////
    // da1
    ierr = DMDAGetLocalInfo(das[0], &info); CHKERRQ(ierr);
    ierr = FormFunctionLocal1(&info, t, u[0], u[1], u_t[0], f[0], p, comp_offset, total_size); CHKERRQ(ierr);
    comp_offset += info.mx;
    ///////////////////////////////////////////////////////
    // da2
    ierr = DMDAGetLocalInfo(das[1], &info); CHKERRQ(ierr);
    ierr = FormFunctionLocal2(&info, t, u[0], u[1], u_t[1], f[1], p, comp_offset, total_size); CHKERRQ(ierr);
    comp_offset += info.mx;
    ///////////////////////////////////////////////////////

    for (int i = 0; i < nDM; ++i) {
        ierr = DMDAVecRestoreArray(das[i], X_ts[i], &(u_t[i])); CHKERRQ(ierr);
        ierr = DMDAVecRestoreArray(das[i], Xs[i], &(u[i])); CHKERRQ(ierr);
        ierr = DMDAVecRestoreArray(das[i], Fs[i], &(f[i])); CHKERRQ(ierr);
    }

    ierr = DMCompositeRestoreAccessArray(dm, F, nDM, NULL, Fs); CHKERRQ(ierr);
    ierr = DMCompositeRestoreLocalVectorsArray(dm, nDM, NULL, Xs); CHKERRQ(ierr);
    ierr = DMCompositeRestoreLocalVectorsArray(dm, nDM, NULL, X_ts); CHKERRQ(ierr);

    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "CompositeSetCoupling"
PetscErrorCode CompositeSetCoupling(DM dm) {
    PetscErrorCode ierr = DMCompositeSetCoupling(dm, FormCoupleLocations);
    return ierr;
}


#undef __FUNCT__
#define __FUNCT__ "FormCoupleLocations"
/*
Computes the coupling between DA1 and DA2. This determines the location of each coupling between DA1 and DA2.
*/
PetscErrorCode FormCoupleLocations(DM dm, Mat A, PetscInt *dnz, PetscInt *onz, PetscInt __rstart, PetscInt __nrows, PetscInt __start, PetscInt __end)
{
    PetscInt       cols[3], row;
    PetscErrorCode ierr;
    PetscInt nDM;

    // TODO: Find a way to dynamically allocate these arrays in c
    DM  das[NETWORK_SIZE];

    PetscFunctionBegin;
   
    ierr = DMCompositeGetNumberDM(dm, &nDM); CHKERRQ(ierr);
    ierr = DMCompositeGetEntriesArray(dm, das); CHKERRQ(ierr);

    DMDALocalInfo  info1, info2;
    ierr = DMDAGetLocalInfo(das[0], &info1); CHKERRQ(ierr);
    ierr = DMDAGetLocalInfo(das[1], &info2); CHKERRQ(ierr);

    // Hack: Bug in petsc file -> packm.c @ line (173) and line (129)
    // First A is NULL, then later dnz and onz are NULL, that's why
    // we need an IF here.
    if (!A) {
        cols[0] = info1.mx;
        row = info1.mx - 1;
        ierr = MatPreallocateLocation(A, row, 1, cols, dnz, onz); CHKERRQ(ierr);
        cols[0] = info1.mx - 1;
        row = info1.mx;
        ierr = MatPreallocateLocation(A, row, 1, cols, dnz, onz); CHKERRQ(ierr);
    }
    else {
        PetscScalar values[1];

        row = info1.mx - 1;
        cols[0] = info1.mx;
        values[0] = 0.0;
        ierr = MatSetValues(A, 1, &row, 1, cols, values, INSERT_VALUES); CHKERRQ(ierr);

        row = info1.mx;
        cols[0] = info1.mx - 1;
        values[0] = 0.0;
        ierr = MatSetValues(A, 1, &row, 1, cols, values, INSERT_VALUES); CHKERRQ(ierr);
    }
    
    PetscFunctionReturn(0);
}
