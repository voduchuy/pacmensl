//
// Created by Huy Vo on 2019-06-29.
//

#include <gtest/gtest.h>
#include "Sys.h"
#include "FspMatrixConstrained.h"
#include "CvodeFsp.h"
#include "KrylovFsp.h"
#include "PetscWrap.h"
#include"pacmensl_test_env.h"

TEST(PetscWrapTest, vec){
  PetscErrorCode ierr;
  pacmensl::Petsc<Vec> v;
  ierr = VecCreate(PETSC_COMM_WORLD, v.mem());
  ASSERT_FALSE(ierr);
  ierr = VecSetSizes(v, 10, PETSC_DECIDE);
  ASSERT_FALSE(ierr);
  ierr = VecSetUp(v);
  ASSERT_FALSE(ierr);
}

TEST(PetscWrapTest, mat){
  PetscErrorCode ierr;
  pacmensl::Petsc<Mat> A;
  ierr = MatCreate(PETSC_COMM_WORLD, A.mem());
  ASSERT_FALSE(ierr);
  ierr = MatSetSizes(A, 10, 10, PETSC_DECIDE, PETSC_DECIDE);
  ASSERT_FALSE(ierr);
  ierr = MatSetType(A, MATAIJ);
  ASSERT_FALSE(ierr);
  ierr = MatSetUp(A);
  ASSERT_FALSE(ierr);
}