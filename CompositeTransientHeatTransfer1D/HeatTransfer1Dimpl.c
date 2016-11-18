#include "HeatTransfer1Dimpl.h"

#undef __FUNCT__
#define __FUNCT__ "FormInitGuess"
/*
FormInitialGuessComp -
Forms the initial guess for the composite model
Unwraps the global solution vector and passes its local pieces into the user functions
*/
PetscErrorCode FormInitGuess(DM dm, Vec X, Params *p)
{
	PetscErrorCode ierr;
	Vec            X1, X2;
	DM             da1, da2;

	PetscFunctionBegin;
	ierr = DMCompositeGetEntries(dm, &da1, &da2); CHKERRQ(ierr);

	/* Access the subvectors in X */
	ierr = DMCompositeGetAccess(dm, X, &X1, &X2); CHKERRQ(ierr);

	/* Evaluate local user provided function */
	ierr = FormInitGuessLocal(da1, X1, p); CHKERRQ(ierr);
	ierr = FormInitGuessLocal(da2, X2, p); CHKERRQ(ierr);

	ierr = DMCompositeRestoreAccess(dm, X, &X1, &X2); CHKERRQ(ierr);
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
				- k * (x2[i].T - x2[i - 1].T) / dx + Q * dx);
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
	Vec            X1, X2;
	Vec            X1_t, X2_t;
	Vec            F1, F2;
    DM             dm;
	DM             da1, da2;

    PetscFunctionBegin;

    ierr = TSGetDM(ts, &dm); CHKERRQ(ierr);
		
	ierr = DMCompositeGetEntries(dm, &da1, &da2); CHKERRQ(ierr);

	/* Access the subvectors in X */
	ierr = DMCompositeGetLocalVectors(dm, &X1, &X2); CHKERRQ(ierr);
	ierr = DMCompositeScatter(dm, X, X1, X2); CHKERRQ(ierr);

	ierr = DMCompositeGetLocalVectors(dm, &X1_t, &X2_t); CHKERRQ(ierr);
	ierr = DMCompositeScatter(dm, X_t, X1_t, X2_t); CHKERRQ(ierr);
	
	PetscInt nDM, idx;
	DMCompositeGetNumberDM(dm, &nDM);

	PetscInt total_size = 0;

	ierr = DMDAGetLocalInfo(da1, &info); CHKERRQ(ierr);
	total_size += info.mx;
	ierr = DMDAGetLocalInfo(da2, &info); CHKERRQ(ierr);
	total_size += info.mx;

	/* Access the subvectors in F.
	These are not ghosted so directly access the memory locations in F */
	ierr = DMCompositeGetAccess(dm, F, &F1, &F2); CHKERRQ(ierr);

    ierr = DMDAVecGetArray(da1, X1, &u1); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da1, X1_t, &u1_t); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da1, F1, &f1); CHKERRQ(ierr);

	ierr = DMDAVecGetArray(da2, X2, &u2); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(da2, X2_t, &u2_t); CHKERRQ(ierr);
	ierr = DMDAVecGetArray(da2, F2, &f2); CHKERRQ(ierr);

	PetscInt comp_offset = 0;

	// Obs for fluxes we need all u1 and u2 !
	///////////////////////////////////////////////////////
	// da1
	ierr = DMDAGetLocalInfo(da1, &info); CHKERRQ(ierr);
    ierr = FormFunctionLocal1(&info, t, u1, u2, u1_t, f1, p, comp_offset, total_size); CHKERRQ(ierr);
	comp_offset += info.mx;
	///////////////////////////////////////////////////////
	// da2
	ierr = DMDAGetLocalInfo(da2, &info); CHKERRQ(ierr);
	ierr = FormFunctionLocal2(&info, t, u1, u2, u2_t, f2, p, comp_offset, total_size); CHKERRQ(ierr);
	comp_offset += info.mx;
	///////////////////////////////////////////////////////

    ierr = DMDAVecRestoreArray(da1, X1, &u1); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da1, X1_t, &u1_t); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(da1, F1, &f1); CHKERRQ(ierr);

	ierr = DMDAVecRestoreArray(da2, X2, &u2); CHKERRQ(ierr);
	ierr = DMDAVecRestoreArray(da2, X2_t, &u2_t); CHKERRQ(ierr);	
	ierr = DMDAVecRestoreArray(da2, F2, &f2); CHKERRQ(ierr);

	ierr = DMCompositeRestoreAccess(dm, F, &F1, &F2); CHKERRQ(ierr);

	ierr = DMCompositeRestoreLocalVectors(dm, &X1, &X2); CHKERRQ(ierr);
	ierr = DMCompositeRestoreLocalVectors(dm, &X1_t, &X2_t); CHKERRQ(ierr);

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
PetscErrorCode FormCoupleLocations(DM dmcomposite, Mat A, PetscInt *dnz, PetscInt *onz, PetscInt __rstart, PetscInt __nrows, PetscInt __start, PetscInt __end)
{
	PetscInt       i, j, cols[3], istart, jstart, in, jn, row, col, M, dim;
	PetscErrorCode ierr;
	DM             da1, da2;

	PetscFunctionBegin;

	// Hack: Bug in petsc file -> packm.c @ line (173)
	if (!dnz) {
		PetscFunctionReturn(0);
	}

	ierr = DMCompositeGetEntries(dmcomposite, &da1, &da2); CHKERRQ(ierr);
	ierr = DMDAGetInfo(da1, 0, &M, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0); CHKERRQ(ierr);
	ierr = DMDAGetCorners(da1, &istart, &jstart, PETSC_NULL, &in, &jn, PETSC_NULL); CHKERRQ(ierr);

	/* coupling from physics 1 to physics 2 */
	row = __rstart;  /* global location of first omega on this process */
	col = __rstart;  /* global location of first temp on this process */

	// HACK!!!!
	cols[0] = 5;
	row = 4;
	ierr = MatPreallocateLocation(A, row, 1, cols, dnz, onz); CHKERRQ(ierr);

	cols[0] = 4;
	row = 5;
	ierr = MatPreallocateLocation(A, row, 1, cols, dnz, onz); CHKERRQ(ierr);


	//M = 10;
	//for (i = 0; i < M; i++) {
	//	if (i == 0) {
	//		cols[0] = row + 1;
	//		cols[1] = row;
	//		ierr = MatPreallocateLocation(A, row, 2, cols, dnz, onz); CHKERRQ(ierr);
	//	}
	//	else if (i == M - 1) {
	//		cols[0] = row - 1;
	//		cols[1] = row;
	//		ierr = MatPreallocateLocation(A, row, 2, cols, dnz, onz); CHKERRQ(ierr);
	//	}
	//	else {
	//		cols[0] = row - 1;
	//		cols[1] = row + 1;
	//		cols[2] = row + 1;
	//		ierr = MatPreallocateLocation(A, row, 3, cols, dnz, onz); CHKERRQ(ierr);
	//	}
	//	row += 1;
	//}

	//for (j = jstart; j<jstart + jn; j++) {
	//	for (i = istart; i<istart + in; i++) {

	//		/* each omega is coupled to the temp to the left and right */
	//		if (i == 0) {
	//			cols[0] = col + 1;
	//			ierr = MatPreallocateLocation(A, row, 1, cols, dnz, onz); CHKERRQ(ierr);
	//		}
	//		else if (i == M - 1) {
	//			cols[0] = col - 1;
	//			ierr = MatPreallocateLocation(A, row, 1, cols, dnz, onz); CHKERRQ(ierr);
	//		}
	//		else {
	//			cols[0] = col - 1;
	//			cols[1] = col + 1;
	//			ierr = MatPreallocateLocation(A, row, 2, cols, dnz, onz); CHKERRQ(ierr);
	//		}
	//		row += 1;
	//		col += 1;
	//	}
	//}

	///* coupling from physics 2 to physics 1 */
	//col = __rstart;  /* global location of first u on this process */
	//row = __rstart;  /* global location of first temp on this process */
	//for (j = jstart; j<jstart + jn; j++) {
	//	for (i = istart; i<istart + in; i++) {

	//		/* temp is coupled to both u and v at each point */
	//		cols[0] = col;
	//		cols[1] = col + 1;
	//		ierr = MatPreallocateLocation(A, row, 2, cols, dnz, onz); CHKERRQ(ierr);
	//		row += 1;
	//		col += 1;
	//	}
	//}

	PetscFunctionReturn(0);
}
