//
// Created by Huy Vo on 5/31/19.
//

#include "StatePartitioner.h"

namespace pacmensl {
    void StatePartitioner::set_up(PartitioningType part_type, PartitioningApproach part_approach) {
        switch (part_type) {
            case GRAPH:
                base = new StatePartitionerGraph(comm);
                break;
            case HYPERGRAPH:
                base = new StatePartitionerHyperGraph(comm);
                break;
            default:
                base = new StatePartitionerBase(comm);
        }
        base->set_lb_approach(part_approach);
    }

    void StatePartitioner::partition(arma::Mat<int> &states, Zoltan_DD_Struct *state_directory,
                                     arma::Mat<int> &stoich_mat,
                                     int *layout) {
        base->partition(states, state_directory, stoich_mat, layout);
    }
}