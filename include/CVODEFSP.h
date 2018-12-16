//
// Created by Huy Vo on 12/6/18.
//

#ifndef PARALLEL_FSP_CVODEFSP_H
#define PARALLEL_FSP_CVODEFSP_H

#include<cvode/cvode.h>
#include<cvode/cvode_spils.h>
#include<sunlinsol/sunlinsol_spbcgs.h>
#include<sunlinsol/sunlinsol_spgmr.h>
#include<sundials/sundials_nvector.h>
#include<nvector/nvector_petsc.h>
#include "FiniteProblemSolver.h"

#ifndef NDEBUG
    #define CVODECHKERR(comm, flag){\
    if (flag < 0) \
    {\
    PetscPrintf(comm, "\nSUNDIALS_ERROR: function failed in file %s line %d with flag = %d\n\n",\
    __FILE__,__LINE__, flag);\
    MPI_Abort(comm, 1);\
    }\
    }
#else
#define CVODECHKERR(comm, flag){while(0){}};
#endif

namespace cme{
    namespace petsc{
        class CVODEFSP : public FiniteProblemSolver{
        protected:
            void* cvode_mem = nullptr;
            SUNLinearSolver linear_solver = nullptr;
            N_Vector solution_wrapper = nullptr;
            N_Vector solution_tmp = nullptr;
            PetscReal t_now_tmp = 0.0;
            PetscReal rel_tol = 1.0e-4;
            PetscReal abs_tol = 1.0e-8;
            int cvode_stat;
        public:
            explicit CVODEFSP(MPI_Comm _comm, int lmm = CV_BDF, int iter = CV_NEWTON);
            void SetCVodeTolerances(PetscReal _r_tol, PetscReal _abs_tol);

            PetscInt Solve() override;
            void Free() override;
            static int cvode_rhs(double t, N_Vector u, N_Vector udot, void *FPS_ptr);
            static int cvode_jac(N_Vector v, N_Vector Jv, realtype t,
                                 N_Vector u, N_Vector fu,
                                 void *FPS_ptr, N_Vector tmp);
            ~CVODEFSP();
        };
    }
}



#endif //PARALLEL_FSP_CVODEFSP_H
