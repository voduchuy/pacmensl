//
// Created by Huy Vo on 5/31/19.
//

#include "StatePartitioner.h"

namespace pecmeal {
    void StatePartitioner::set_up(PartitioningType part_type, PartitioningApproach part_approach) {
        switch (part_type) {
            case Graph:
                base = new StatePartitionerGraph(comm);
                break;
            case HyperGraph:
                base = new StatePartitionerHyperGraph(comm);
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