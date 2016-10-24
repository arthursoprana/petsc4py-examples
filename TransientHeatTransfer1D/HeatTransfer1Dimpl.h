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

PetscErrorCode FormInitGuess(DM da, Vec x, Params *p);
PetscErrorCode FormFunction(TS ts, PetscReal t, Vec X, Vec X_t, Vec F, Params *p);
PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, PetscReal t, Field *x, Field *x_t, Field *f, Params *p);

#endif /* !HEATTRANSFER1D_H */
