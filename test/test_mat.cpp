//
// Created by Huy Vo on 12/4/18.
//
//
// Created by Huy Vo on 12/3/18.
//
static char help[] = "Test the generation of the distributed Finite State Subset for the toggle model.\n\n";

#include<petsc.h>
#include<petscvec.h>
#include<petscmat.h>
#include<petscao.h>
#include<armadillo>
#include"models/toggle_model.h"
#include"models/repressilator_model.h"
#include"util/cme_util.h"
#include"FSS/FiniteStateSubsetBase.h"
#include"Matrix/FspMatrixBase.h"

using namespace cme::parallel;

int main(int argc, char *argv[]) {
    int ierr;
    arma::Row<double> fsp_size = {3.0, 3.0};

    ierr = PetscInitialize(&argc, &argv, (char *) 0, help);
    CHKERRQ(ierr);
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    // Read options for fsp
    char opt[100];
    PetscBool opt_set;
    PartitioningType fsp_par_type = Graph;
    ierr = PetscOptionsGetString(NULL, PETSC_NULL, "-fsp_partitioning_type", opt, 100, &opt_set);
    CHKERRQ(ierr);
    if (opt_set) {
        fsp_par_type = str2part(std::string(opt));
        PetscPrintf(PETSC_COMM_WORLD, "Partitioning with option %s \n", opt);
    }

    double Q_sum;
    {
//        arma::Mat<PetscInt> X0(2, 1); X0.fill(0);
//        StateSetBase fsp(PETSC_COMM_WORLD, 2);
//        fsp.set_lb_type(fsp_par_type);
//        fsp.set_shape_bounds(fsp_size);
//        fsp.set_stoichiometry(toggle_cme::SM);
//        fsp.set_initial_states(X0);
//        fsp.expand();
//        PetscPrintf(PETSC_COMM_WORLD, "State Subset generated.\n");
//
//        FspMatrixBase A(PETSC_COMM_WORLD);
//        A.GenerateMatrices(fsp, toggle_cme::SM, toggle_cme::propensity, toggle_cme::t_fun);

        arma::Mat<PetscInt> X0(3, 1); X0.fill(0); X0(0) = 20;
        FiniteStateSubsetBase fsp(PETSC_COMM_WORLD, 3);
        fsp.set_lb_type( fsp_par_type );
        fsp.set_shape( &repressilator_cme::lhs_constr, repressilator_cme::rhs_constr );
        fsp.set_stoichiometry( repressilator_cme::SM );
        fsp.set_initial_states( X0 );
        fsp.expand( );
        PetscPrintf(PETSC_COMM_WORLD, "State Subset generated.\n");

        FspMatrixBase A(PETSC_COMM_WORLD);
        A.GenerateMatrices(fsp, repressilator_cme::SM, repressilator_cme::propensity, repressilator_cme::t_fun);

        Vec P, Q;
        VecCreate(PETSC_COMM_WORLD, &P);
        VecSetSizes(P, fsp.get_num_local_states( ) + fsp.get_shape_bounds( ).n_elem, PETSC_DECIDE);
        VecSetFromOptions(P);
        VecSet(P, 1.0);
        VecSetUp(P);
        VecDuplicate(P, &Q);
        VecSetUp(Q);

        A.Action(1.0, P, Q);
        VecView(Q, PETSC_VIEWER_STDOUT_WORLD);


        VecSum(Q, &Q_sum);
        PetscPrintf(PETSC_COMM_WORLD, "Q_sum = %.2f \n", Q_sum);

        A.Destroy();
    }
    ierr = PetscFinalize();
    CHKERRQ(ierr);
//    assert(std::abs(Q_sum) <= 1.0e-14);
    return 0;
}
