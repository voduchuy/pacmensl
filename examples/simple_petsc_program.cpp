//
// Created by Huy Vo on 2019-08-05.
//
#include <petsc.h>

int main(int argc, char *argv[])
{
  PetscInitialize(&argc, &argv, 0, 0);

  Vec x;

  VecCreate(PETSC_COMM_WORLD, &x);
  VecSetSizes(x, 100, PETSC_DETERMINE);
  VecSetUp(x);
  VecSet(x, 1.0);

  PetscFinalize();
}
