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
#include"util/cme_util.h"
#include"FSS/StateSetConstrained.h"
#include"Matrix/FspMatrixConstrained.h"

using namespace cme::parallel;

int main(int argc, char *argv[]) {
    int ierr;

    arma::Row<int> fsp_size({10});
    double rate_right{2.0},rate_left{3.0};

    arma::Mat<int> stoichiometry({1,-1});
    auto t_fun = [&] (double t){
        return arma::Row<double>({1.0, 1.0});
    };

    auto propensity = [&] (const PetscInt *X, const PetscInt k) {
        switch (k){
            case 0:
                return rate_right;
            case 1:
                return rate_left*(X[0]>0);
            default:
                return 0.0;
        }
    };

    ierr = PetscInitialize(&argc, &argv, (char *) 0, help);
    CHKERRQ(ierr);
    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    int comm_size;
    MPI_Comm_size(PETSC_COMM_WORLD, &comm_size);

    // Read options for state_set_
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
        arma::Mat<PetscInt> X0(1, 1); X0.fill(0); X0(0) = 0;
        StateSetConstrained fsp(PETSC_COMM_WORLD, 1, fsp_par_type);
        fsp.set_stoichiometry( stoichiometry );
        fsp.set_shape_bounds(fsp_size);
        fsp.set_initial_states( X0 );
        fsp.expand( );
        PetscPrintf(PETSC_COMM_WORLD, "State Subset generated.\n");

        FspMatrixConstrained A(PETSC_COMM_WORLD);
        A.generate_matrices( fsp, stoichiometry, propensity, t_fun );

        Vec P, Q;
        VecCreate(PETSC_COMM_WORLD, &P);
        int i = (rank == comm_size-1)? 1 : 0;
        VecSetSizes(P, fsp.get_num_local_states( ) + i, PETSC_DECIDE);
        VecSetFromOptions(P);
        VecSet(P, 1.0);
        VecSetUp(P);
        VecDuplicate(P, &Q);
        VecSetUp(Q);

        A.action( 1.0, P, Q );
        VecView(Q, PETSC_VIEWER_STDOUT_WORLD);


        VecSum(Q, &Q_sum);
        PetscPrintf(PETSC_COMM_WORLD, "Q_sum = %.2f \n", Q_sum);

        A.destroy( );
    }
    ierr = PetscFinalize();
    CHKERRQ(ierr);
    if (Q_sum != 0.0) return -1;
    return 0;
}
