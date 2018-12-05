static char help[] = "Solve the 5-species spatial hog1p model with time-varying propensities.\n\n";

#include <petscmat.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <petscts.h>
#include <armadillo>
#include <cmath>
#include <sundials/sundials_nvector.h>
#include <nvector/nvector_petsc.h>
#include <cvode/cvode.h>
#include <cvode/cvode_spils.h>
#include <sunlinsol/sunlinsol_spbcgs.h>
#include "MatrixSet.h"
#include "models/hog1p_tv_model.h"

using arma::dvec;
using arma::Col;
using arma::Row;

using std::cout;
using std::endl;

void petscvec_to_file(MPI_Comm comm, Vec x, const char *filename);

void petscmat_to_file(MPI_Comm comm, Mat A, const char *filename);

using namespace hog1p_cme;

/* RHS of CME routine. */
static int cvode_rhs(double t, N_Vector u, N_Vector udot, void* FSPMat_ptr){
    Vec* udata = N_VGetVector_Petsc(u);
    Vec* udotdata = N_VGetVector_Petsc(udot);
    ((cme::petsc::MatrixSet*) FSPMat_ptr)->set_time(t);
    ((cme::petsc::MatrixSet*) FSPMat_ptr)->action(*udata, *udotdata);
    return 0;
}

/* Jacobian-times-vector routine. */
static int cvode_jac(N_Vector v, N_Vector Jv, realtype t,
                     N_Vector u, N_Vector fu,
                     void *FSPMat_ptr, N_Vector tmp)
{
    Vec* vdata = N_VGetVector_Petsc(v);
    Vec* Jvdata = N_VGetVector_Petsc(Jv);
    ((cme::petsc::MatrixSet*) FSPMat_ptr)->set_time(t);
    ((cme::petsc::MatrixSet*) FSPMat_ptr)->action(*vdata, *Jvdata);
    return 0;
}

static int check_flag(void *flagvalue, const char *funcname, int opt);

int main(int argc, char *argv[]) {
    int ierr, myRank, num_procs;
    double tic, solver_time;
    std::string model_name = "hog1p";

    int cvode_flag;
    double rel_tol = 1.0e-4, abs_tol = 1.0e-8;

    /* CME problem sizes */
    Row<PetscInt> FSPSize({3, 63, 63, 63, 63}); // Size of the FSP
    PetscInt n_species = 5;
    PetscInt n_states = arma::prod(FSPSize + 1);
    PetscReal t_final = 200.0;

    ierr = PetscInitialize(&argc, &argv, (char *) 0, help);
    CHKERRQ(ierr);

    MPI_Comm comm{PETSC_COMM_WORLD};

    MPI_Comm_size(comm, &num_procs);

    cme::petsc::MatrixSet A(comm, FSPSize, SM, propensity, t_fun);

    /* Test the action of the operator */
    Vec P0, P, P_rk;
    VecCreate(comm, &P0);
    VecCreate(comm, &P);

    VecSetSizes(P0, PETSC_DECIDE, n_states + n_species);
    VecSetType(P0, VECMPI);
    VecSetUp(P0);
    VecSet(P0, 0.0);
    VecSetValue(P0, 0, 1.0, INSERT_VALUES);
    VecAssemblyBegin(P0);
    VecAssemblyEnd(P0);
    VecDuplicate(P0, &P);
    VecCopy(P0, P);


    N_Vector P_cvode = N_VMake_Petsc(&P);
    void *cvode_mem = CVodeCreate(CV_BDF, CV_NEWTON);
    cvode_flag = CVodeInit(cvode_mem, cvode_rhs, 0.0, P_cvode);
    if(check_flag(&cvode_flag, "CVodeInit", 1)) return(1);

    /* Call CVodeSStolerances to specify the scalar relative tolerance
    * and scalar absolute tolerance */
    cvode_flag = CVodeSStolerances(cvode_mem, rel_tol, abs_tol);
    if (check_flag(&cvode_flag, "CVodeSStolerances", 1)) return (1);

    /* Set the pointer to user-defined data */
    cvode_flag = CVodeSetUserData(cvode_mem, (void *) &A);
    if (check_flag(&cvode_flag, "CVodeSetUserData", 1)) return (1);

    cvode_flag = CVodeSetMaxNumSteps(cvode_mem, 10000000);
    cvode_flag = CVodeSetMaxConvFails(cvode_mem, 10000000);
    cvode_flag = CVodeSetStabLimDet(cvode_mem, 1);
    cvode_flag = CVodeSetMaxNonlinIters(cvode_mem, 100000);

    /* Create SPGMR solver structure without preconditioning
    * and the maximum Krylov dimension maxl */
    SUNLinearSolver LS = SUNSPBCGS(P_cvode, PREC_NONE, 0);
    if (check_flag(&cvode_flag, "SUNSPBCGS", 1)) return (1);

    /* Set CVSpils linear solver to LS */
    cvode_flag = CVSpilsSetLinearSolver(cvode_mem, LS);
    if (check_flag(&cvode_flag, "CVSpilsSetLinearSolver", 1)) return (1);

    /* Set the JAcobian-times-vector function */
    cvode_flag = CVSpilsSetJacTimes(cvode_mem, NULL, cvode_jac);
    if (check_flag(&cvode_flag, "CVSpilsSetJacTimesVecFn", 1)) return (1);

    tic = MPI_Wtime();
    PetscReal t = 0.0;
    PetscReal psum = 0.0;
    Vec *P_cvode_dat = N_VGetVector_Petsc(P_cvode);
    while (t < t_final){
        cvode_flag = CVode(cvode_mem, t_final, P_cvode, &t, CV_ONE_STEP);
        if(check_flag(&cvode_flag, "CVode", 1)) {
            std::cout << "cvode_flag = " << cvode_flag << "\n";
            break;
        }
        ierr = VecSum(*P_cvode_dat, &psum); CHKERRQ(ierr);
        std::cout << "t = " << t << " psum = " << psum << "\n";
    }
    solver_time = MPI_Wtime() - tic;

    /* Compute the marginal distributions */
    std::vector<arma::Col<PetscReal>> marginals(FSPSize.n_elem);
    for (PetscInt i{0}; i < marginals.size(); ++i) {
        marginals[i] = cme::petsc::marginal(*P_cvode_dat, FSPSize, i);
    }

    MPI_Comm_rank(comm, &myRank);
    if (myRank == 0) {
        {
            std::string filename = model_name + "_time_" + std::to_string(num_procs) + ".dat";
            std::ofstream file;
            file.open(filename);
            file << solver_time;
            file.close();
        }
        for (PetscInt i{0}; i < marginals.size(); ++i) {
            std::string filename =
                    model_name + "_marginal_" + std::to_string(i) + "_" + std::to_string(num_procs) + ".dat";
            marginals[i].save(filename, arma::raw_ascii);
        }
    }

    N_VDestroy(P_cvode);
    ierr = VecDestroy(&P);
    CHKERRQ(ierr);
    ierr = VecDestroy(&P0);
    CHKERRQ(ierr);
    A.destroy();
    ierr = PetscFinalize();
    return ierr;
}

void petscvec_to_file(MPI_Comm comm, Vec x, const char *filename) {
    PetscViewer viewer;
    PetscViewerCreate(comm, &viewer);
    PetscViewerBinaryOpen(comm, filename, FILE_MODE_WRITE, &viewer);
    VecView(x, viewer);
    PetscViewerDestroy(&viewer);
}

void petscmat_to_file(MPI_Comm comm, Mat A, const char *filename) {
    PetscViewer viewer;
    PetscViewerCreate(comm, &viewer);
    PetscViewerBinaryOpen(comm, filename, FILE_MODE_WRITE, &viewer);
    MatView(A, viewer);
    PetscViewerDestroy(&viewer);
}


/* Check function return value...
     opt == 0 means SUNDIALS function allocates memory so check if
              returned NULL pointer
     opt == 1 means SUNDIALS function returns a flag so check if
              flag >= 0
     opt == 2 means function allocates memory so check if returned
              NULL pointer */

static int check_flag(void *flagvalue, const char *funcname, int opt)
{
    int *errflag;

    /* Check if SUNDIALS function returned NULL pointer - no memory allocated */

    if (opt == 0 && flagvalue == NULL) {
        fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
                funcname);
        return(1); }

        /* Check if flag < 0 */

    else if (opt == 1) {
        errflag = (int *) flagvalue;
        if (*errflag < 0) {
            fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with flag = %d\n\n",
                    funcname, *errflag);
            return(1); }}

        /* Check if function returned NULL pointer - no memory allocated */

    else if (opt == 2 && flagvalue == NULL) {
        fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
                funcname);
        return(1); }

    return(0);
}