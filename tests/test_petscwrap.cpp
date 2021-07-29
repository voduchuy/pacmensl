/*
MIT License

Copyright (c) 2020 Huy Vo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

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