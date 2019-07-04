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

  PacmenslErrorCode SetConstraintFunctions(const fsp_constr_multi_fn &lhs_constr);

  PacmenslErrorCode SetInitialBounds(arma::Row<int> &_fsp_size);

  PacmenslErrorCode SetExpansionFactors(arma::Row<PetscReal> &_expansion_factors);

  PacmenslErrorCode SetModel(Model &model);

  PacmenslErrorCode SetVerbosity(int verbosity_level);

  PacmenslErrorCode SetUp();

  PacmenslErrorCode SetInitialDistribution(const arma::Mat<Int> &_init_states, const arma::Col<PetscReal> &_init_probs);

  PacmenslErrorCode SetLoadBalancingMethod(PartitioningType part_type);

  DiscreteDistribution Solve(PetscReal sfsp_tol);

  PacmenslErrorCode ClearState();

  ~StationaryFspSolverMultiSinks();

 protected:
  MPI_Comm comm_ = nullptr;
  int my_rank_;
  int comm_size_;

  bool set_up_ = false;

  int  verbosity_               = 0;

  PartitioningType partitioning_type_ = PartitioningType::GRAPH;
  PartitioningApproach repart_approach_ = PartitioningApproach::REPARTITION;

  bool have_custom_constraints_ = false;
  fsp_constr_multi_fn fsp_constr_funs_;
  arma::Row<int>      fsp_bounds_;
  arma::Row<Real>     fsp_expasion_factors_;

  arma::Row<PetscReal> sinks_;
  arma::Row<int>       to_expand_;

  arma::Mat<Int>       init_states_;
  arma::Col<PetscReal> init_probs_;

  TIMatvec matvec_;

  Model model_;

  Vec solution_ = nullptr;
  std::unique_ptr<StateSetConstrained> state_set_;
  std::unique_ptr<StationaryFspMatrixConstrained> matrix_;
  std::unique_ptr<StationaryMCSolver> solver_;

  PacmenslErrorCode MakeDiscreteDistribution_(DiscreteDistribution& dist);
};
}
#endif //PACMENSL_SRC_STATIONARYFSP_STATIONARYFSPSOLVERMULTISINKS_H_
