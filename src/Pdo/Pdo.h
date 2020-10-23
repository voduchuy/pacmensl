//
// Created by Huy Vo on 10/23/20.
//

#ifndef PACMENSL_SRC_PDO_PDO_H_
#define PACMENSL_SRC_PDO_PDO_H_

#include <mpi.h>
#include "StateSet.h"
#include "DiscreteDistribution.h"
#include "SensDiscreteDistribution.h"


namespace pacmensl{
  typedef std::function<int (int n, int xdim, int* x, int ydim, int* y, double* pyx)> TkFun;
  /**
   * @brief Probabilistic Distortion Operator, a class for transforming a distribution of a discrete multivariate variable X into
   * the distribution of another variable Y, using the conditional probability P(Y|X).
   */
  class Pdo {
   public:
    
    explicit Pdo(MPI_Comm comm, int source_dim, int target_dim);
    
    DiscreteDistribution Transform(DiscreteDistribution p);
    SensDiscreteDistribution Transform(SensDiscreteDistribution p);
    
   protected:
    MPI_Comm comm_ = MPI_COMM_NULL;
    TkFun transition_fun_ = nullptr;
    int source_dim_ = 0;
    int target_dim_ = 0;
  };
}

#endif //PACMENSL_SRC_PDO_PDO_H_
