//
// Created by Huy Vo on 5/31/19.
//

#include "StatePartitioner.h"

namespace pacmensl {
    void StatePartitioner::SetUp(PartitioningType part_type, PartitioningApproach part_approach) {
        switch (part_type) {
            case PartitioningType::GRAPH:
                data = new StatePartitionerGraph(comm);
                break;
            case PartitioningType::HYPERGRAPH:
                data = new StatePartitionerHyperGraph(comm);
                break;
            default:
                data = new StatePartitionerBase(comm);
        }
        data->set_lb_approach(part_approach);
    }

    int StatePartitioner::Partition(arma::Mat<int> &states, Zoltan_DD_Struct *state_directory,
                                    arma::Mat<int> &stoich_mat,
                                    int *layout) {
        return data->partition(states, state_directory, stoich_mat, layout);
    }
}