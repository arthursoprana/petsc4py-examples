#ifndef HEATTRANSFER1D_H
#define HEATTRANSFER1D_H

#include <petsc.h>

#include <petsc/private/vecimpl.h> // This MUST BE INCLUDED FOR DEBUG REASONS!
#include <petsc/private/matimpl.h> // This MUST BE INCLUDED FOR DEBUG REASONS!
#include <petsc/private/dmimpl.h> // This MUST BE INCLUDED FOR DEBUG REASONS!

typedef struct Params {
  double temperature_left_, temperature_right_, conductivity_;
} Params;

typedef struct {
  PetscScalar T;
} Field;

PetscErrorCode FormInitGuess(DM da, Vec x, Params *p);
PetscErrorCode FormFunction(DM da, Vec x, Vec F, Params *p);
PetscErrorCode FormFunctionLocal(DMDALocalInfo *info,Field *x,Field *f,Params *p);

#endif /* !HEATTRANSFER1D_H */
