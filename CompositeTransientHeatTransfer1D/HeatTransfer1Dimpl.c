#include "HeatTransfer1Dimpl.h"
#include "include_patch_pack.h"

#define NODES_SIZE 1
#define PIPES_SIZE 2
#define DMS_SIZE 3


#undef __FUNCT__
#define __FUNCT__ "RedundantSetSize"
PetscErrorCode RedundantSetSize(DM dm, PetscMPIInt rank, PetscInt N) {
    PetscErrorCode ierr;
    ierr = DMRedundantSetSize(dm, rank, N); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

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
    PetscInt nDM;
    // TODO: Find a way to dynamically allocate these arrays in c
    DM  das[DMS_SIZE];
    Vec Xs[DMS_SIZE];
   
    PetscFunctionBegin;

    ierr = DMCompositeGetNumberDM(dm, &nDM); CHKERRQ(ierr);
    ierr = DMCompositeGetEntriesArray(dm, das); CHKERRQ(ierr);

    /* Access the subvectors in X */
    ierr = DMCompositeGetAccessArray(dm, X, nDM, NULL, Xs); CHKERRQ(ierr);

    /* Evaluate local user provided function */
    for (int i = 0; i < nDM - NODES_SIZE; ++i) {
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
#define __FUNCT__ "FormFunctionLocal"
PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, PetscReal t, Field *x, Field *x_t, Field *f, Params *p, PetscInt comp_offset, PetscInt total_size)
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
         
    // Not sure why we need this :(
    if (xints == 0) {
        i = 0;
        xints = xints + 1;
        /* Temperature */
        f[i].T = x_t[i].T * dx
               -1.0 * (+ k * (x[i + 1].T - x[i].T) / dx + Q * dx);
    }
    if (xinte == info->mx) {
        i = info->mx - 1;
        xinte = xinte - 1;  
        /* Temperature */
        f[i].T = x_t[i].T * dx
               -1.0 * (- k * (x[i].T - x[i - 1].T) / dx + Q * dx);      
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
    PetscErrorCode ierr;
    DM             dm;
    PetscInt nDM;

    // TODO: Find a way to dynamically allocate these arrays in c
    DM  das[DMS_SIZE];
    Vec Xs[DMS_SIZE], X_ts[DMS_SIZE], Fs[DMS_SIZE];
    Field* u_t[DMS_SIZE];
    Field* u[DMS_SIZE];
    Field* f[DMS_SIZE];

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
    for (int i = 0; i < nDM - NODES_SIZE; ++i) {
        ierr = DMDAGetLocalInfo(das[i], &info); CHKERRQ(ierr);
        total_size += info.mx;
    }
	total_size += 1;

    /* Access the subvectors in F.
    These are not ghosted so directly access the memory locations in F */
    ierr = DMCompositeGetAccessArray(dm, F, nDM, NULL, Fs); CHKERRQ(ierr);

    PetscInt comp_offset = 0;
    for (int i = 0; i < nDM - NODES_SIZE; ++i) {
        ierr = DMDAVecGetArray(das[i], X_ts[i], &(u_t[i])); CHKERRQ(ierr);
        ierr = DMDAVecGetArray(das[i], Xs[i], &(u[i])); CHKERRQ(ierr);
        ierr = DMDAVecGetArray(das[i], Fs[i], &(f[i])); CHKERRQ(ierr);
        
        ierr = DMDAGetLocalInfo(das[i], &info); CHKERRQ(ierr);
        
        // Central nodes
        ierr = FormFunctionLocal(&info, t, u[i], u_t[i], f[i], p, comp_offset, total_size); CHKERRQ(ierr);
        comp_offset += info.mx;
    }

	// Nodal part of residual
    ierr = VecGetArray(X_ts[PIPES_SIZE], &(u_t[PIPES_SIZE])); CHKERRQ(ierr);
    ierr = VecGetArray(Xs[PIPES_SIZE], &(u[PIPES_SIZE])); CHKERRQ(ierr);
    ierr = VecGetArray(Fs[PIPES_SIZE], &(f[PIPES_SIZE])); CHKERRQ(ierr);

    // Code goes here
	{
		PetscReal      dx;
		PetscReal      L, k, conductivity, T_left, T_right, Q;

		conductivity = p->conductivity_;
		T_left = p->temperature_left_;
		T_right = p->temperature_right_;
		Q = p->source_term_;
		L = p->wall_length_;
		dx = L / (PetscReal)(total_size - 1);
		k = conductivity;

		// Internal node
		double acc_term = u_t[PIPES_SIZE][0].T * dx;
		double source_term = Q * dx;
		f[PIPES_SIZE][0].T = acc_term - source_term;

		ierr = DMDAGetLocalInfo(das[0], &info); CHKERRQ(ierr);
		double flux_term_L = k * (u[PIPES_SIZE][0].T - u[0][info.mx-1].T) / dx;
		double flux_term_R = k * (u[1][0].T - u[PIPES_SIZE][0].T) / dx;
		f[PIPES_SIZE][0].T += -(flux_term_R - flux_term_L);

		f[0][info.mx - 1].T += +(-flux_term_L);
		f[1][0].T           += +(+flux_term_R);

		// bc's
		// left flux term w/ prescribed pressure
		f[0][0].T += - 1.0 * (-k * (u[0][0].T - T_left) / (0.5*dx));

		ierr = DMDAGetLocalInfo(das[1], &info); CHKERRQ(ierr);
		f[1][info.mx - 1].T += -1.0 * (+k * (T_right - u[1][info.mx - 1].T) / (0.5*dx));
	}
    // end
    ierr = VecRestoreArray(X_ts[PIPES_SIZE], &(u_t[PIPES_SIZE])); CHKERRQ(ierr);
    ierr = VecRestoreArray(Xs[PIPES_SIZE], &(u[PIPES_SIZE])); CHKERRQ(ierr);
    ierr = VecRestoreArray(Fs[PIPES_SIZE], &(f[PIPES_SIZE])); CHKERRQ(ierr);

    for (int i = 0; i < nDM - NODES_SIZE; ++i) {
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
    DMDALocalInfo info;
    // TODO: Find a way to dynamically allocate these arrays in c
    DM  das[DMS_SIZE];

    PetscFunctionBegin;
   
    ierr = DMCompositeGetNumberDM(dm, &nDM); CHKERRQ(ierr);
    ierr = DMCompositeGetEntriesArray(dm, das); CHKERRQ(ierr);

    // Calculate total size
    int size = 0;
    for (int i = 0; i < nDM - NODES_SIZE; ++i) {
        ierr = DMDAGetLocalInfo(das[i], &info); CHKERRQ(ierr);
        size += info.mx;
    }

	for (int i = nDM - NODES_SIZE; i < nDM; ++i) {
		PetscInt N;
		ierr = DMRedundantGetSize(das[i], NULL, &N); CHKERRQ(ierr);
		size += N;
	}


    // Hack: Bug in petsc file -> packm.c @ line (173) and line (129)
    // First A is NULL, then later dnz and onz are NULL, that's why
    // we need an IF here.
    if (!A) {
		int next_pipe_start_idx = 0;
		for (int i = 0; i < nDM - NODES_SIZE - 1; ++i) {
			ierr = DMDAGetLocalInfo(das[i], &info); CHKERRQ(ierr);
			next_pipe_start_idx += info.mx;

			cols[0] = next_pipe_start_idx - 1;
			row = size - 1;
			ierr = MatPreallocateLocation(A, row, 1, cols, dnz, onz); CHKERRQ(ierr);
			cols[0] = next_pipe_start_idx;
			row = size - 1;
			ierr = MatPreallocateLocation(A, row, 1, cols, dnz, onz); CHKERRQ(ierr);

			cols[0] = size - 1;
			row = next_pipe_start_idx - 1;
			ierr = MatPreallocateLocation(A, row, 1, cols, dnz, onz); CHKERRQ(ierr);
			cols[0] = size - 1;
			row = next_pipe_start_idx;
			ierr = MatPreallocateLocation(A, row, 1, cols, dnz, onz); CHKERRQ(ierr);
		}
    }
    else {
        PetscScalar values[1];
		int next_pipe_start_idx = 0;
		for (int i = 0; i < nDM - NODES_SIZE - 1; ++i) {
			ierr = DMDAGetLocalInfo(das[i], &info); CHKERRQ(ierr);
			next_pipe_start_idx += info.mx;

			cols[0] = next_pipe_start_idx - 1;
			row = size - 1;
			values[0] = 0.0;
			ierr = MatSetValues(A, 1, &row, 1, cols, values, INSERT_VALUES); CHKERRQ(ierr);
			cols[0] = next_pipe_start_idx;
			row = size - 1;
			values[0] = 0.0;
			ierr = MatSetValues(A, 1, &row, 1, cols, values, INSERT_VALUES); CHKERRQ(ierr);

			cols[0] = size - 1;
			row = next_pipe_start_idx - 1;
			values[0] = 0.0;
			ierr = MatSetValues(A, 1, &row, 1, cols, values, INSERT_VALUES); CHKERRQ(ierr);
			cols[0] = size - 1;
			row = next_pipe_start_idx;
			values[0] = 0.0;
			ierr = MatSetValues(A, 1, &row, 1, cols, values, INSERT_VALUES); CHKERRQ(ierr);
		}
    }
    PetscFunctionReturn(0);
}
