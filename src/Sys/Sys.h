#ifndef PACMENSL_UTIL_H
#define PACMENSL_UTIL_H

#define ARMA_DONT_PRINT_ERRORS
#define ARMA_DONT_USE_WRAPPER
#include <armadillo>
#include <petscmat.h>
#include <petscvec.h>
#include <petscao.h>
#include <petscis.h>
#include <petscistypes.h>
#include <petscoptions.h>
#include <petsc.h>
#include <petscsys.h>
#include <petscconf.h>
#include <cassert>
#include <memory>
#include <mpi.h>
#include <zoltan.h>
#include <parmetis.h>
#include <string>
#include <sstream>
#include "ErrorHandling.h"

#define NOT_COPYABLE_NOT_MOVABLE(object)\
            object( const object & ) = delete;\
            object &operator=( const object & ) = delete;\

namespace pacmensl {
#define SQR1 sqrt(0.1e0)

/*! Round to 2 significant digits
 */
double round2digit(double x);

/*! Initialize and finalize Parallel context
 *
 */
int PACMENSLInit(int *argc, char ***argv, const char *help);

int PACMENSLFinalize();

void sequential_action(MPI_Comm comm, std::function<void(void *)> action, void *data);

class Environment {
 public:
  Environment();

  Environment(int *argc, char ***argv, const char *help);

  ~Environment();

 private:
  bool initialized = false;
  bool init_petsc = false; // If PETSc or MPI were already set, we do not meddle with them
  bool init_mpi = false;
};
}

#endif