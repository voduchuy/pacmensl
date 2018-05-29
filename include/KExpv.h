#pragma once
#include <vector>
#include <armadillo>
#include <petscmat.h>
#include <petscvec.h>

namespace cme {
namespace petsc {
using Real = PetscReal;
using Int = PetscInt;
using MVFun = std::function<void (Vec x, Vec y)>;

/**
 * @brief Wrapper class for Krylov-based evaluation of the matrix exponential.
 * @details Compute the expression exp(t_f*A)*v, where A and v are PETSC matrix and vector objects. The matrix does not enter explicitly but through the matrix-vector product. The Krylov-based approximation used is based on Sidje's Expokit, with the option to use Incomplete Orthogonalization Process in place of the full Arnoldi.
 */
class KExpv {

protected:

    const MPI_Comm comm = MPI_COMM_NULL;        ///< Commmunicator for the Krylov algorithm. This must be the same as the one used for the vector and matrix involved.

MVFun matvec;
Vec solution_now = nullptr;

Real t_final;       ///< Final time/scaling of the matrix.

Int i_step = 0;

Real t_now;
Real t_new;
Real t_step;

const Real delta = 1.2;       ///< Safety factor in stepsize adaptivity
const Real gamma = 0.9;       ///< Safety factor in stepsize adaptivity

Real btol;

std::vector<Vec> V;       ///< Container for the Krylov vectors
arma::Mat<Real> H;       ///< The small, dense Hessenberg matrix, each process in comm owns a copy of this matrix
Vec av;
arma::Mat<Real> F;

public:
Real tol = 1.0e-8;
Real anorm = 1.0;

bool IOP = false;         ///< Flag for using incomplete orthogonalization. (default false)
Int q_iop = 2;         ///< IOP parameter, the current Krylov vector will be orthogonalized against q_iop-1 previous ones

Int m = 30;         ///< Size of the Krylov subspace for each step
Int max_nstep = 10000;
Int max_reject = 1000;

/**
 * @brief Constructor for KExpv without initializing vector data structures
 * @details User must manually call update_vectors method before using the object.
 */
    KExpv(MPI_Comm _comm, Real _t_final, Int _m, Real _tol = 1.0e-8, bool _iop = false, Int _q_iop = 2, Real _anorm = 1.0) :
            comm(_comm),
            t_final(_t_final),
            m(_m),
            tol(_tol),
            IOP(_iop),
            q_iop(_q_iop),
            anorm(_anorm),
            t_now(0.0)
    {
    }

/**
 * @brief Constructor for KExpv with vector data structures.
 */
KExpv(MPI_Comm _comm, Real _t_final, MVFun _matvec, Vec _v, Int _m, Real _tol = 1.0e-8, bool _iop = false, Int _q_iop = 2, Real _anorm = 1.0) :
        comm(_comm),
        t_final(_t_final),
        m(_m),
        tol(_tol),
        IOP(_iop),
        q_iop(_q_iop),
        anorm(_anorm),
        t_now(0.0)
{
        update_vectors(_v, _matvec);
}

/**
 * @brief Set a new final time and reset the current time to 0.
 * @details The current solution vector will be kept, so solve() will integrate a new linear system with the current solution vector as the initial solution.
 */
void reset_time(Real _t_final)
{
  t_now = 0.0;
  t_final = _t_final;
}

/**
 * @brief Set a new initial solution and reset the current time to 0.
 * @details The new initial solution must have the same distributed structure as the old solution.
 */
void reset_init_sol(Vec new_initial_solution)
{
  t_now = 0.0;
  solution_now = new_initial_solution;
}

/**
 * @brief Integrate all the way to t_final.
 */
void solve();

/**
 * @brief Advance to the furthest time possible using a Krylov basis of max dimension m.
 */
void step();

/**
 * @brief Check if the final time has been reached.
 */
bool final_time_reached()
{
        return (t_now >= t_final);
}

/**
 * @brief Update the vector data structure
 */
 void update_vectors(Vec _v, MVFun _matvec)
    {
        Int ierr;
        matvec = _matvec;
        solution_now = _v;
        if (av == PETSC_NULL)
        {
            V.resize(m+1);
            if (comm == MPI_COMM_NULL || comm == nullptr)
            {
                std::cout << "Null comm passed!\n";
            }
            /* Duplicate the distributed structure of the intial vector to the Krylov vectors */
            for (Int i{0}; i< m+1; ++i) {
                ierr = VecCreate(comm, &V[i]); CHKERRABORT(comm, ierr);
                ierr = VecDuplicate(solution_now, &V[i]); CHKERRABORT(comm, ierr);
            }
            ierr = VecCreate(comm, &av); CHKERRABORT(comm, ierr);
            ierr = VecDuplicate(solution_now, &av); CHKERRABORT(comm, ierr);
        }
    }

/**
 * @brief Free the resources for the internal Krylov vectors.
 * @details This must be called when the object is no longer needed. TODO: automatic destruction.
 */
void destroy()
{
        Int ierr;
        for (Int i{0}; i< m+1; ++i) {
                ierr = VecDestroy(&V[i]); CHKERRABORT(comm, ierr);
        }
        ierr = VecDestroy(&av); CHKERRABORT(comm, ierr);
}

};
}
}
