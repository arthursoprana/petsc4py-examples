#include <petsc.h>

static char help[] = "Implementation of a non-linear fieldsplit";

PETSC_EXTERN PetscErrorCode SNESFieldSplitSetRuntimeSplits_Private(SNES);


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
    SNES snes;
    PetscErrorCode ierr;
    ierr = PetscInitialize(&argc, &argv, (char*)0, help); if (ierr) return(1);

    ierr = SNESFieldSplitSetRuntimeSplits_Private(snes); CHKERRQ(ierr);
    ierr = PetscFinalize();
    return 0;
}

#undef __FUNCT__
#define __FUNCT__ "SNESCreate_FieldSplit"
/*PETSC_EXTERN*/ PetscErrorCode SNESCreate_FieldSplit(SNES snes)
{
    SNES_FieldSplit  *jac;
    PetscErrorCode   ierr;

    PetscFunctionBegin;
    snes->ops->destroy = SNESDestroy_FieldSplit;
    //snes->ops->setup          = SNESSetUp_FieldSplit;
    //snes->ops->setfromoptions = SNESSetFromOptions_FieldSplit;
    //snes->ops->view           = SNESView_FieldSplit;
    //snes->ops->solve          = SNESSolve_FieldSplit;
    snes->ops->reset = SNESReset_FieldSplit;

    snes->usesksp = PETSC_FALSE;

    ierr = PetscNewLog(snes, &jac); CHKERRQ(ierr);
    jac->bs = -1;
    jac->nsplits = 0;
    jac->type = PC_COMPOSITE_MULTIPLICATIVE;
    jac->schurpre = PC_FIELDSPLIT_SCHUR_PRE_USER; /* Try user preconditioner first, fall back on diagonal */
    jac->schurfactorization = PC_FIELDSPLIT_SCHUR_FACT_FULL;
    jac->dm_splits = PETSC_TRUE;

    snes->data = (void*)jac;

    //ierr = PetscObjectComposeFunction((PetscObject)snes,"SNESFieldSplitGetSubSNES_C", SNESFieldSplitGetSubSNES_FieldSplit);CHKERRQ(ierr);
    //ierr = PetscObjectComposeFunction((PetscObject)snes,"SNESFieldSplitSetFields_C", SNESFieldSplitSetFields_FieldSplit);CHKERRQ(ierr);
    //ierr = PetscObjectComposeFunction((PetscObject)snes,"SNESFieldSplitSetIS_C", SNESFieldSplitSetIS_FieldSplit);CHKERRQ(ierr);
    //ierr = PetscObjectComposeFunction((PetscObject)snes,"SNESFieldSplitSetType_C", SNESFieldSplitSetType_FieldSplit);CHKERRQ(ierr);
    //ierr = PetscObjectComposeFunction((PetscObject)snes,"SNESFieldSplitSetBlockSize_C", SNESFieldSplitSetBlockSize_FieldSplit);CHKERRQ(ierr);
    //ierr = PetscObjectComposeFunction((PetscObject)snes,"SNESFieldSplitRestrictIS_C", SNESFieldSplitRestrictIS_FieldSplit);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
