//
// Created by Huy Vo on 2019-06-29.
//

#ifndef PACMENSL_SRC_PETSCWRAP_PETSCWRAP_H_
#define PACMENSL_SRC_PETSCWRAP_PETSCWRAP_H_
#include <petsc.h>
namespace pacmensl{

template <typename PetscT>
class Petsc{
 protected:
  PetscT dat = nullptr;
 public:

  Petsc(){
    static_assert(std::is_convertible<PetscT, Vec>::value || std::is_convertible<PetscT, Mat>::value
    || std::is_convertible<PetscT, IS>::value || std::is_convertible<PetscT, VecScatter>::value,
    "pacmensl::Petsc can only wrap PETSc objects.");
  }

  PetscT* mem() {return &dat;}

  const PetscT* mem() const {return &dat;}

  bool IsEmpty() { return (dat==nullptr);}

  operator PetscT() { return dat;}

  int Destroy(PetscT* obj);

  ~Petsc(){
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

}
#endif //PACMENSL_SRC_PETSCWRAP_PETSCWRAP_H_
