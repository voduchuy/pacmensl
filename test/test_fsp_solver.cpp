//
// Created by Huy Vo on 12/6/18.
//

//
// Created by Huy Vo on 12/6/18.
//
static char help[] = "Test interface to CVODE for solving the CME of the toggle model.\n\n";

#include "pecmeal_all.h"
#include "FspSolverBase.h"
#include "toggle_model.h"

using namespace pecmeal;

int main(int argc, char *argv[]) {
    std::string model_name = "toggle_switch";

    PetscInt ierr;
    PetscReal t_final = 100.0, fsp_tol = 1.0e-2;
    arma::Mat<PetscInt> X0 = {0,0}; X0 = X0.t();
    arma::Col<PetscReal> p0 = {1.0};

    ierr = PetscInitialize(&argc, &argv, (char *) 0, help);
    CHKERRQ(ierr);

    // Get processor rank and number of processors
    PetscMPIInt rank, num_procs;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &num_procs);

    // Read options for fsp
    PartitioningType fsp_par_type = Graph;
    ODESolverType fsp_odes_type = CVODE_BDF;
    char opt[100];
    PetscBool opt_set;
    ierr = PetscOptionsGetString(NULL, PETSC_NULL, "-fsp_partitioning_type", opt, 100, &opt_set);
    fsp_par_type = str2part(std::string(opt));
    PetscPrintf(PETSC_COMM_WORLD, "FSP is partitioned with %s.\n", part2str(fsp_par_type).c_str());

    // Begin PETSC context
    {
        Model toggle_model(toggle_cme::SM, toggle_cme::t_fun, toggle_cme::propensity);
        arma::Row<int> fsp_size = {3, 3};
        arma::Row<PetscReal> expansion_factors = {0.25,0.25};
        FspSolverBase fsp(PETSC_COMM_WORLD, fsp_par_type, fsp_odes_type);
        fsp.SetModel(toggle_model);
        fsp.SetInitialBounds(fsp_size);
        fsp.SetExpansionFactors(expansion_factors);
        fsp.SetVerbosity(2);
        fsp.SetInitialDistribution(X0, p0);
        fsp.SetUp();
        DiscreteDistribution p_final = fsp.Solve(t_final, fsp_tol);
        fsp.Destroy();

        fsp.SetModel(toggle_model);
        fsp.SetInitialBounds(fsp_size);
        fsp.SetExpansionFactors(expansion_factors);
        fsp.SetVerbosity(2);
        fsp.SetInitialDistribution(X0, p0);
        fsp.SetUp();
        arma::Row<PetscReal> tspan = arma::linspace<arma::Row<PetscReal>>(0.0, 100.0, 10);
        std::vector<DiscreteDistribution> p_snapshots = fsp.Solve(tspan, fsp_tol);
        fsp.Destroy();
    }
    //End PETSC context
    ierr = PetscFinalize();
    CHKERRQ(ierr);
    return 0;
}