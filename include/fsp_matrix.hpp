#pragma once
#include <petscmat.h>
#include "cme_util.hpp"

using PropFun = std::function<arma::Mat<double> (arma::Mat<PetscInt>)> ;

void FSPMatSetValues(MPI_Comm comm, Mat &A, arma::Row<PetscInt> &FSPSize, arma::Mat<int> SM,  PropFun propensity);
