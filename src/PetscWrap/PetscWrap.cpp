//
// Created by Huy Vo on 2019-06-29.
//

#include "PetscWrap.h"

namespace pacmensl {
template<>
int Petsc<Vec>::Destroy(Vec *obj) { return VecDestroy(obj); }

template<>
int Petsc<Mat>::Destroy(Mat *obj) { return MatDestroy(obj); }

template<>
int Petsc<IS>::Destroy(IS *obj) { return ISDestroy(obj); }

template<>
int Petsc<VecScatter>::Destroy(VecScatter *obj) { return VecScatterDestroy(obj); }

template<>
int Petsc<KSP>::Destroy(KSP *obj) { return KSPDestroy(obj);}

template<>
int Petsc<TS>::Destroy(TS *obj) {return TSDestroy(obj);};

PacmenslErrorCode ExpandVec(Vec &p,
                            const std::vector<PetscInt> &new_indices,
                            const PetscInt new_local_size)
{
  PacmenslErrorCode ierr{0};
  Petsc<IS>         new_locations;
  Petsc<VecScatter> scatter;
  MPI_Comm          comm;

  comm = PetscObjectComm(( PetscObject ) p);

  Vec Pnew;
  ierr = VecCreate(comm, &Pnew); PACMENSLCHKERRQ(ierr);
  ierr = VecSetSizes(Pnew, new_local_size, PETSC_DECIDE); PACMENSLCHKERRQ(ierr);
  ierr = VecSetUp(Pnew); PACMENSLCHKERRQ(ierr);
  ierr = VecSet(Pnew, 0.0); PACMENSLCHKERRQ(ierr);
  ierr = VecAssemblyBegin(Pnew); PACMENSLCHKERRQ(ierr);
  ierr = VecAssemblyEnd(Pnew); PACMENSLCHKERRQ(ierr);

  // Scatter from old vector to the expanded vector
  ierr = ISCreateGeneral(comm, ( PetscInt ) new_indices.size(), new_indices.data(),
                         PETSC_USE_POINTER, new_locations.mem()); PACMENSLCHKERRQ(ierr);
  ierr = VecScatterCreate(p, PETSC_IGNORE, Pnew, new_locations, scatter.mem()); PACMENSLCHKERRQ(ierr);
  ierr = VecScatterBegin(scatter, p, Pnew, INSERT_VALUES, SCATTER_FORWARD); PACMENSLCHKERRQ(ierr);
  ierr = VecScatterEnd(scatter, p, Pnew, INSERT_VALUES, SCATTER_FORWARD); PACMENSLCHKERRQ(ierr);

  // Swap p_ to the expanded vector
  ierr = VecDestroy(&p); PACMENSLCHKERRQ(ierr);
  p    = Pnew;
  return 0;
}

PacmenslErrorCode ExpandVec(Petsc<Vec> &p,
                                      const std::vector<PetscInt> &new_indices,
                                      const PetscInt new_local_size)
{
  return ExpandVec(*p.mem(), new_indices, new_local_size);
}
}

