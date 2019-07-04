//
// Created by Huy Vo on 2019-06-29.
//

#ifndef PACMENSL_SRC_PETSCWRAP_PETSCWRAP_H_
#define PACMENSL_SRC_PETSCWRAP_PETSCWRAP_H_
#include <petsc.h>
#include <iostream>
#include "Sys.h"
namespace pacmensl {

template<typename PetscT>
class Petsc {
 protected:
  PetscT dat = PETSC_NULL;

  int Destroy(PetscT *obj);

 public:

  Petsc() {
    static_assert(std::is_convertible<PetscT, Vec>::value || std::is_convertible<PetscT, Mat>::value
                      || std::is_convertible<PetscT, IS>::value || std::is_convertible<PetscT, VecScatter>::value
                      || std::is_convertible<PetscT, KSP>::value,
                  "pacmensl::Petsc can only wrap PETSc objects.");
  }

  PetscT *mem() { return &dat; }

  const PetscT *mem() const { return &dat; }

  bool IsEmpty() { return (dat == nullptr); }

  operator PetscT() { return dat; }

  ~Petsc() {
    Destroy(&dat);
  }
};

template<>
int Petsc<Vec>::Destroy(Vec *obj);

template<>
int Petsc<Mat>::Destroy(Mat *obj);

template<>
int Petsc<IS>::Destroy(IS *obj);

template<>
int Petsc<VecScatter>::Destroy(VecScatter *obj);

template<>
int Petsc<KSP>::Destroy(KSP *obj);

PacmenslErrorCode ExpandVec(Petsc<Vec> &p, const std::vector<PetscInt> &new_indices, const PetscInt new_local_size);
PacmenslErrorCode ExpandVec(Vec &p, const std::vector<PetscInt> &new_indices, const PetscInt new_local_size);
}
#endif //PACMENSL_SRC_PETSCWRAP_PETSCWRAP_H_
