//
// Created by Huy Vo on 12/12/18.
//

static char help[] = "Generate Finite State Subset and output to files.\n\n";

#include<iomanip>
#include<memory.h>
#include <petscmat.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <util/cme_util.h>
#include <armadillo>
#include <cmath>
#include "FSP/FSPSolver.h"
#include "models/toggle_model.h"
#include "models/hog1p_5d_model.h"
#include "models/transcription_regulation_6d_model.h"
#include "models/hog1p_3d_model.h"

using arma::dvec;
using arma::Col;
using arma::Row;

using std::cout;
using std::endl;

using namespace hog1p_cme;
using namespace cme::parallel;

int main(int argc, char *argv[]) {

    PetscMPIInt myRank, num_procs;
    PetscErrorCode ierr;
    std::string model_name, part_option;
    arma::Row<PetscInt> FSPSize;
    arma::Mat<PetscInt> X0;
    arma::Mat<PetscInt> stoich_mat;
    PartitioningType fsp_par_type;
    std::fstream ofs;
    std::string filename;
    arma::Mat<PetscInt> local_states;

    ierr = PetscInitialize(&argc, &argv, (char *) 0, help);
    CHKERRQ(ierr);
    // Begin PETSC context
    {
        MPI_Comm comm;
        MPI_Comm_dup(PETSC_COMM_WORLD, &comm);
        MPI_Comm_size(comm, &num_procs);
        MPI_Comm_rank(comm, &myRank);
        PetscPrintf(comm, "\n ================ \n");

        // Default options
        model_name = "toggle";
        FSPSize = {90, 60}; // Size of the FSP
        stoich_mat = toggle_cme::SM;
        part_option = "graph";
        fsp_par_type = Graph;
        X0.set_size(2, 1); X0.fill(0);

        // Read options for fsp
        char opt[100];
        PetscBool opt_set;

        ierr = PetscOptionsGetString(NULL, PETSC_NULL, "-fsp_model", opt, 100, &opt_set);
        CHKERRQ(ierr);
        if (opt_set) {
            if (strcmp(opt, "transcr_reg_6d") == 0) {
                model_name = "transcr_reg_6d";
                FSPSize = {10, 6, 1, 2, 1, 1}; // Size of the FSP
                X0.set_size(6, 1); X0.fill(0);
                stoich_mat = six_species_cme::SM;
                PetscPrintf(PETSC_COMM_WORLD, "Problem: Transcription regulation with 6 species.\n");
            } else if (strcmp(opt, "hog5d") == 0) {
                model_name = "hog1p";
                FSPSize = {3, 3, 3, 3, 3}; // Size of the FSP
                X0.set_size(5, 1); X0.fill(0);
                stoich_mat = hog1p_cme::SM;
                PetscPrintf(PETSC_COMM_WORLD, "Problem: Hog1p with 5 species.\n");
            } else if (strcmp(opt, "hog3d") == 0) {
                model_name = "hog3d";
                FSPSize = {3, 10, 10}; // Size of the FSP
                X0.set_size(3, 1); X0.fill(0);
                stoich_mat = hog3d_cme::SM;
                PetscPrintf(PETSC_COMM_WORLD, "Problem: Hog1p with 3 species.\n");
            } else {
                PetscPrintf(PETSC_COMM_WORLD, "Problem: Toggle-switch.\n");
            }
        }
        else{
            PetscPrintf(PETSC_COMM_WORLD, "Problem: Toggle-switch.\n");
        }

        ierr = PetscOptionsGetString(NULL, PETSC_NULL, "-fsp_partitioning_type", opt, 100, &opt_set);
        CHKERRQ(ierr);
        if (opt_set) {
            fsp_par_type = str2part(opt);
            part_option = part2str(fsp_par_type);
            PetscPrintf(PETSC_COMM_WORLD, "FSP is partitionined with %s. \n", part_option.c_str());
        }
        else{
            PetscPrintf(PETSC_COMM_WORLD, "FSP is partitioned with Graph.\n");
        }
        PetscPrintf(comm, "Distributing to %d processors.\n", num_procs);

        FiniteStateSubset state_set(comm, X0.n_rows);
        state_set.SetStoichiometry(stoich_mat);
        state_set.SetShapeBounds(FSPSize);
        state_set.SetInitialStates(X0);
        state_set.SetLBType(fsp_par_type);

        state_set.GenerateStatesAndOrdering();

        local_states = state_set.GetLocalStates();
        local_states = local_states.t();

        filename = model_name + "_local_states_" + std::to_string(myRank) + "_of_" + std::to_string(num_procs) + "_" + part_option + ".dat";
        local_states.save(filename, arma::raw_ascii);

        PetscPrintf(comm, "\n ================ \n");
    }
    //End PETSC context
    ierr = PetscFinalize();
    return ierr;
}

