/*
MIT License

Copyright (c) 2020 Huy Vo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

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