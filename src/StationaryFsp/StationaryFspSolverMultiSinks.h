//
// Created by Huy Vo on 2019-06-25.
//

#ifndef PACMENSL_SRC_STATIONARYFSP_STATIONARYFSPSOLVERMULTISINKS_H_
#define PACMENSL_SRC_STATIONARYFSP_STATIONARYFSPSOLVERMULTISINKS_H_

#include "DiscreteDistribution.h"
#include "StationaryFspMatrixConstrained.h"
#include "StationaryMCSolver.h"

namespace pacmensl {
class StationaryFspSolverMultiSinks {
 public:
  explicit StationaryFspSolverMultiSinks(MPI_Comm comm);

  int SetConstraintFunctions(const fsp_constr_multi_fn &lhs_constr);

  int SetInitialBounds(arma::Row<int> &_fsp_size);

  int SetExpansionFactors(arma::Row<PetscReal> &_expansion_factors);

  int SetModel(Model &model);

  int SetVerbosity(int verbosity_level);

  int SetUp();

  int SetInitialDistribution(const arma::Mat<Int> &_init_states, const arma::Col<PetscReal> &_init_probs);

  int SetLogging(PetscBool logging);

  int SetLoadBalancingMethod(PartitioningType part_type);

  DiscreteDistribution Solve(PetscReal sfsp_tol);

  int ClearState();

  ~StationaryFspSolverMultiSinks();

 protected:
  MPI_Comm comm_ = nullptr;
  int my_rank_;
  int comm_size_;

  PartitioningType partitioning_type_ = PartitioningType::GRAPH;
  PartitioningApproach repart_approach_ = PartitioningApproach::REPARTITION;

  Model model_;

  PetscReal tol_;
  StateSetConstrained state_set_;
  StationaryFspMatrixConstrained matrix_;
  StationaryMCSolver solver_;
  Vec solution_;
};
}
#endif //PACMENSL_SRC_STATIONARYFSP_STATIONARYFSPSOLVERMULTISINKS_H_
