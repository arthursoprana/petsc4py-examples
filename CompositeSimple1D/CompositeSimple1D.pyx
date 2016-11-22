
from petsc4py.PETSc cimport Vec,  PetscVec
from petsc4py.PETSc cimport DM,   PetscDM
from petsc4py.PETSc cimport SNES, PetscSNES
from petsc4py.PETSc cimport TS,   PetscTS

from petsc4py.PETSc import Error
import cython
import numpy as np
cimport numpy as np

cdef extern from "CompositeSimple1Dimpl.h":
    ctypedef struct Params:
        double* temperature_presc_
        double conductivity_
        double source_term_
        double wall_length_        
    
    int FormFunction (PetscTS ts, double t, PetscVec X, PetscVec X_t, PetscVec F, Params *p)    
    int CompositeSetCoupling (PetscDM da)
    int RedundantSetSize (PetscDM dm, int rank, int N)

@cython.boundscheck(False)
@cython.wraparound(False)
def formFunction(TS ts, double t, Vec x, Vec x_t, Vec f, 
    double conductivity_, 
    double source_term_, 
    double wall_length_,
    np.ndarray[double, ndim=1, mode="c"] temperature_presc_ not None, 
    ):
    cdef int ierr
    cdef Params p = {
        "temperature_presc_" : &temperature_presc_[0], 
        "conductivity_" : conductivity_,
        "source_term_" : source_term_,
        "wall_length_" : wall_length_,
    }
    ierr = FormFunction(ts.ts, t, x.vec, x_t.vec, f.vec, &p)
    if ierr != 0: raise Error(ierr)
    
def compositeSetCoupling(DM dm):
    cdef int ierr
    ierr = CompositeSetCoupling(dm.dm)
    if ierr != 0: raise Error(ierr)
    
def redundantSetSize(DM dm, int rank, int N):
    cdef int ierr
    ierr = RedundantSetSize(dm.dm, rank, N)
    if ierr != 0: raise Error(ierr)
