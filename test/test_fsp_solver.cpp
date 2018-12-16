//
// Created by Huy Vo on 12/6/18.
//

//
// Created by Huy Vo on 12/6/18.
//
static char help[] = "Test interface to CVODE for solving the CME of the toggle model.\n\n";

#include "cme_util.h"
#include "FSPSolver.h"
#include "models/toggle_model.h"

using namespace cme::petsc;

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
    PartioningType fsp_par_type = Naive;
    ODESolverType fsp_odes_type = CVODE_BDF;
    char opt[100];
    PetscBool opt_set;
    ierr = PetscOptionsGetString(NULL, PETSC_NULL, "-fsp_partitioning_type", opt, 100, &opt_set);
    if (opt_set){
        if (strcmp(opt, "Graph") == 0){
            fsp_par_type = Graph;
            PetscPrintf(PETSC_COMM_WORLD, "FSP is partitioned with Graph.\n");
        }
    }

    // Begin PETSC context
    {
        arma::Row<PetscInt> fsp_size = {30, 30};
        arma::Row<PetscReal> expansion_factors = {0.25,0.25};
        FSPSolver fsp(PETSC_COMM_WORLD, fsp_par_type, fsp_odes_type);
        fsp.SetInitFSPSize(fsp_size);
        fsp.SetFSPTolerance(fsp_tol);
        fsp.SetFinalTime(t_final);
        fsp.SetStoichiometry(toggle_cme::SM);
        fsp.SetTimeFunc(toggle_cme::t_fun);
        fsp.SetPropensity(toggle_cme::propensity);
        fsp.SetExpansionFactors(expansion_factors);
        fsp.SetVerbosityLevel(2);
        fsp.SetInitProbabilities(X0, p0);
        fsp.SetUp();

        fsp.Solve();

        /* Compute the marginal distributions */
        Vec P = fsp.GetP();
        FiniteStateSubset& state_set = fsp.GetStateSubset();
        std::vector<arma::Col<PetscReal>> marginals(fsp_size.n_elem);
        for (PetscInt i{0}; i < marginals.size(); ++i) {
            marginals[i] = cme::petsc::marginal(state_set, P, i);
        }

        MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
        if (rank == 0) {
            for (PetscInt i{0}; i < marginals.size(); ++i) {
                std::string filename =
                        model_name + "_marginal_" + std::to_string(i) + "_" + std::to_string(num_procs) + ".dat";
                marginals[i].save(filename, arma::raw_ascii);
            }
        }
    }
    //End PETSC context
    ierr = PetscFinalize();
    CHKERRQ(ierr);
    return 0;
}