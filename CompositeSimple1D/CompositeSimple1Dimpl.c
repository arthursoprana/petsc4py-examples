#include "CompositeSimple1Dimpl.h"
#include "include_patch_pack.h"

#define NODES_SIZE 1
#define PIPES_SIZE 2
#define DMS_SIZE PIPES_SIZE + NODES_SIZE


#include <petsc/private/snesimpl.h>     /*I  "petscsnes.h"  I*/

PETSC_EXTERN PetscErrorCode SNESCreate_Multiblock(SNES);
#undef __FUNCT__
#define __FUNCT__ "RegisterNewSNES"
PetscErrorCode RegisterNewSNES() {
    PetscErrorCode ierr;
    ierr = SNESRegister("multiblock", SNESCreate_Multiblock); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RedundantSetSize"
PetscErrorCode RedundantSetSize(DM dm, PetscMPIInt rank, PetscInt N) {
    PetscErrorCode ierr;
    ierr = DMRedundantSetSize(dm, rank, N); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormFunctionLocal"
PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, PetscReal t, Field *x, Field *x_t, Field *f, Params *p, PetscInt comp_offset, PetscInt total_size)
 {
    PetscErrorCode ierr;
    PetscInt       xints,xinte,i;
    PetscReal      dx, L, k, conductivity, Q;

    PetscFunctionBegin;

    conductivity = p->conductivity_;  
    Q = p->source_term_;
    L = p->wall_length_;

    /* 
     Define mesh intervals ratios for uniform grid.
    */
	dx = L / (PetscReal)(info->mx - 1);
    k = conductivity;

    xints = info->xs; xinte = info->xs+info->xm;
         
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
		PetscReal dx, L, k, conductivity,  Q;

		conductivity = p->conductivity_;
		Q = p->source_term_;
		L = p->wall_length_;
		k = conductivity;

		// Internal node
		f[PIPES_SIZE][0].T = 0.0; 
		double dx_avg = 0.0;

		for (int i = 0; i < PIPES_SIZE; ++i) {
			ierr = DMDAGetLocalInfo(das[i], &info); CHKERRQ(ierr);
			int pipe_end = info.mx - 1;

			dx = L / (PetscReal)(info.mx - 1);

			double flux_term = k * (u[PIPES_SIZE][0].T - u[i][pipe_end].T) / dx;
			f[PIPES_SIZE][0].T += +flux_term;
			f[i][pipe_end].T   += -flux_term;

			dx_avg += dx;
		}

		dx_avg /= PIPES_SIZE;

		double acc_term = u_t[PIPES_SIZE][0].T * dx_avg;
		double source_term = Q * dx_avg;
		f[PIPES_SIZE][0].T += acc_term - source_term;

		// bc's
		// left flux term w/ prescribed pressure
		for (int i = 0; i < PIPES_SIZE; ++i) {
			ierr = DMDAGetLocalInfo(das[i], &info); CHKERRQ(ierr);
			dx = L / (PetscReal)(info.mx - 1);

			double Tpresc = p->temperature_presc_[i];			

			int pipe_begin = 0;
			f[i][pipe_begin].T += -1.0 * (-k * (u[i][pipe_begin].T - Tpresc) / (0.5*dx));
		}
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
	PetscScalar values[1];
    PetscErrorCode ierr;
    PetscInt nDM;
    DMDALocalInfo info;
    // TODO: Find a way to dynamically allocate these arrays in c
    DM  dms[DMS_SIZE];

    PetscFunctionBegin;
   
    ierr = DMCompositeGetNumberDM(dm, &nDM); CHKERRQ(ierr);
    ierr = DMCompositeGetEntriesArray(dm, dms); CHKERRQ(ierr);

    // Calculate total size
    int size = 0;
    for (int i = 0; i < nDM - NODES_SIZE; ++i) {
        ierr = DMDAGetLocalInfo(dms[i], &info); CHKERRQ(ierr);
        size += info.mx * info.dof;
    }

	for (int k = 0; k < NODES_SIZE; ++k) {
		PetscInt N;
		ierr = DMRedundantGetSize(dms[nDM - NODES_SIZE + k], NULL, &N); CHKERRQ(ierr);

		int next_pipe_start_idx = 0;
		for (int i = 0; i < nDM - NODES_SIZE; ++i) {
			ierr = DMDAGetLocalInfo(dms[i], &info); CHKERRQ(ierr);
			next_pipe_start_idx += info.mx * info.dof;

			for (int j = 0; j < info.dof; ++j) {
				for (int h = 0; h < info.dof; ++h) {
					values[0] = 0.0;

					cols[0] = j + next_pipe_start_idx - 1 * info.dof;
					row = k * N + size + h;

					// Hack: Bug in petsc file -> packm.c @ line (173) and line (129)
					// First A is NULL, then later dnz and onz are NULL, that's why
					// we need an IF here.
					if (!A) {
						ierr = MatPreallocateLocation(A, row, 1, cols, dnz, onz); CHKERRQ(ierr);
					}
					else {
						ierr = MatSetValues(A, 1, &row, 1, cols, values, INSERT_VALUES); CHKERRQ(ierr);
					}

					cols[0] = k * N + size + h;
					row = j + next_pipe_start_idx - 1 * info.dof;

					// Hack: Bug in petsc file -> packm.c @ line (173) and line (129)
					// First A is NULL, then later dnz and onz are NULL, that's why
					// we need an IF here.
					if (!A) {
						ierr = MatPreallocateLocation(A, row, 1, cols, dnz, onz); CHKERRQ(ierr);
					}
					else {
						ierr = MatSetValues(A, 1, &row, 1, cols, values, INSERT_VALUES); CHKERRQ(ierr);
					}
				}
			}
		}
	}
    PetscFunctionReturn(0);
}
