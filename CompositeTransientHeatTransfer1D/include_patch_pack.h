#ifndef INCLUDEPATCHPACK_H
#define INCLUDEPATCHPACK_H

#include <petsc.h>
#include "D:\Work\debug-sources\esss-petsc-9e4b98e1cce2\src\dm\impls\composite\packimpl.h"
#undef __FUNCT__
#define __FUNCT__ "DMCompositeGetLocalVectorsArray"
PetscErrorCode  DMCompositeGetLocalVectorsArray(DM dm, PetscInt nwanted, const PetscInt *wanted, Vec *vecs)
{
    PetscErrorCode         ierr;
    struct DMCompositeLink *link;
    PetscInt               i, wnum;
    DM_Composite           *com = (DM_Composite*)dm->data;

    PetscFunctionBegin;
    PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
    for (i = 0, wnum = 0, link = com->next; link && wnum < nwanted; i++, link = link->next) {
        if (!wanted || i == wanted[wnum]) {
            Vec v;
            ierr = DMGetLocalVector(link->dm, &v); CHKERRQ(ierr);
            vecs[wnum++] = v;
        }
    }
    PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMCompositeRestoreLocalVectorsArray"
PetscErrorCode  DMCompositeRestoreLocalVectorsArray(DM dm, PetscInt nwanted, const PetscInt *wanted, Vec *vecs)
{
    PetscErrorCode         ierr;
    struct DMCompositeLink *link;
    PetscInt               i, wnum;
    DM_Composite           *com = (DM_Composite*)dm->data;

    PetscFunctionBegin;
    PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
    for (i = 0, wnum = 0, link = com->next; link && wnum < nwanted; i++, link = link->next) {
        if (!wanted || i == wanted[wnum]) {
            ierr = DMRestoreLocalVector(link->dm, &vecs[wnum]); CHKERRQ(ierr);
            wnum++;
        }
    }
    PetscFunctionReturn(0);
}

#endif /* !INCLUDEPATCHPACK_H */