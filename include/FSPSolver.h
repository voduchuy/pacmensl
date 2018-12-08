//
// Created by Huy Vo on 5/29/18.
//

#ifndef PARALLEL_FSP_FSP_H
#define PARALLEL_FSP_FSP_H

#include<algorithm>
#include<cstdlib>
#include<cmath>
#include"MatrixSet.h"
#include"FiniteProblemSolver.h"
#include"FiniteStateSubset.h"
#include"FiniteStateSubsetLinear.h"
#include"FiniteStateSubsetParMetis.h"
#include"CVODEFSP.h"
#include"cme_util.h"

namespace cme{
    namespace petsc{
        class FSPSolver {
            using Real = PetscReal;
            using Int = PetscInt;
        private:

            MPI_Comm comm = MPI_COMM_NULL;

            PartioningType partioning_type;
            ODESolverType odes_type;

            arma::Row<Int> fsp_size;
            arma::Row<Real> fsp_expasion_factors;

            FiniteStateSubset *fsp;
            Vec *p;
            MatrixSet *A;
            FiniteProblemSolver *ode_solver;

            Real t_final = 0.0;
            Real fsp_tol = 0.0;

            arma::Mat<Int> stoich_mat;
            PropFun propensity;
            TcoefFun t_fun;

            std::function<void (PetscReal, Vec, Vec)> tmatvec;
            arma::Mat<Int> init_states;

            arma::Col<PetscReal> init_probs;
            int verbosity = 0;
        public:

            explicit FSPSolver(MPI_Comm _comm, PartioningType _part_type, ODESolverType _solve_type);

            void SetInitFSPSize(arma::Row<Int> &_fsp_size);
            void SetFinalTime(PetscReal t);
            void SetExpansionFactors(arma::Row<PetscReal> &_expansion_factors);
            void SetFSPTolerance(PetscReal _fsp_tol);
            void SetStoichiometry(arma::Mat<Int> &stoich);
            void SetPropensity(PropFun _prop);
            void SetTimeFunc(TcoefFun _t_fun);
            void SetVerbosityLevel(int verbosity_level);
            void SetInitProbabilities(arma::Mat<Int> &_init_states, arma::Col<PetscReal> &_init_probs);

            void SetUp();

            Vec& GetP();
            FiniteStateSubset& GetStateSubset();

            void Solve();

            void Destroy();

            ~FSPSolver();

            friend FiniteStateSubset;
            friend MatrixSet;
            friend FiniteProblemSolver;
        };
    }
}


#endif //PARALLEL_FSP_FSP_H
