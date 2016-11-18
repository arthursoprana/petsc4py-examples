#ifndef HEATTRANSFER1D_H
#define HEATTRANSFER1D_H

#include <petsc.h>

#include <petsc/private/vecimpl.h> // This MUST BE INCLUDED FOR DEBUG REASONS!
#include <petsc/private/matimpl.h> // This MUST BE INCLUDED FOR DEBUG REASONS!
#include <petsc/private/dmimpl.h> // This MUST BE INCLUDED FOR DEBUG REASONS!
#include <petsc/private/tsimpl.h> // This MUST BE INCLUDED FOR DEBUG REASONS!

typedef struct Params {
    double temperature_left_;
    double temperature_right_;
    double conductivity_;
    double source_term_;
    double wall_length_;
} Params;

typedef struct {
  PetscScalar T;
} Field;

PetscErrorCode FormInitGuess(DM dm, Vec X, Params *p);
PetscErrorCode FormInitGuessLocal(DM da, Vec X, Params *p);
PetscErrorCode FormFunction(TS ts, PetscReal t, Vec X, Vec X_t, Vec F, Params *p);
PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, PetscReal t, Field *x1, Field *x2, Field *x_t, Field *f, Params *p, PetscInt comp_offset, PetscInt total_size);
PetscErrorCode FormCoupleLocations(DM dmcomposite, Mat A, PetscInt *dnz, PetscInt *onz, PetscInt __rstart, PetscInt __nrows, PetscInt __start, PetscInt __end);
PetscErrorCode CompositeSetCoupling(DM dm);

#endif /* !HEATTRANSFER1D_H */
