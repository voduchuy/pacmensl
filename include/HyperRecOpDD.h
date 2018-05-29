#pragma once
#include <iostream>
#include <sstream>
#include <algorithm>
#include <stdlib.h>
#include <cassert>
#include <cmath>
#include <armadillo>
#include <petscmat.h>
#include <petscao.h>
#include "cme_util.h"
#include "mpi.h"

namespace cme
{
namespace petsc
{
using PropFun = std::function< PetscReal (PetscInt *, PetscInt)>;
using TcoefFun = std::function<arma::Row<PetscReal> (PetscReal t)>;
using Real = PetscReal;
using Int = PetscInt;
/* Distributed data type for the truncated CME operator on a hyper-rectangle,
   using domain decomposition.
 */
class HyperRecOpDD
{

protected:
/* The processor group that owns this operator */
MPI_Comm comm {MPI_COMM_NULL};

arma::Row<Int> max_num_molecules;

arma::Row<Int> sub_domain_lb, sub_domain_ub;

Real t_here = 0.0;

std::vector<Mat> terms;

Vec work; ///< Work vector for computing operator times vector

TcoefFun t_fun = NULL;

arma::Mat<Int> local_state_space;
/* Helper to generate the application ordering */
void get_ordering(const arma::Row<Int> &nmax, const arma::Row<Int> &processor_grid, const std::vector<arma::Row<Int> > &sub_domain_dims);

public:
AO ao;
Int n_reactions;
Int n_rows_global;
Int n_rows_here;

/* constructors */
HyperRecOpDD ( MPI_Comm& new_comm, const arma::Row<Int> &new_nmax, const arma::Row<Int> &processor_grid, const std::vector<arma::Row<Int> > sub_domain_dims, const arma::Mat<Int> &SM, PropFun prop, TcoefFun new_t_fun);

/* Set current time for the matrix */
void set_time( Real t_in );

HyperRecOpDD& operator()(Real t)
{
        set_time(t);
        return *this;
}

void destroy()
{
        for (PetscInt i{0}; i< n_reactions+1; ++i)
        {
                MatDestroy(&terms[i]);
        }
        VecDestroy(&work);
        AODestroy(&ao);
}

void duplicate_structure(Mat &A);
void dump_to_mat(Mat A);

void print_info();

void action(Vec x, Vec y);
};
}
}
