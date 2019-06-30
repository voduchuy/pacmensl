//
// Created by Huy Vo on 2019-06-29.
//

#include "PetscWrap.h"

namespace pacmensl{
template<>
int Petsc<Vec>::Destroy(Vec *obj){return VecDestroy(obj);}

template<>
int Petsc<Mat>::Destroy(Mat *obj){return MatDestroy(obj);}

template<>
int Petsc<IS>::Destroy(IS *obj){return ISDestroy(obj);}

template<>
int Petsc<VecScatter>::Destroy(VecScatter *obj){return VecScatterDestroy(obj);}
}