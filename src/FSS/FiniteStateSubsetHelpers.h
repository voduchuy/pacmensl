//
// Created by Huy Vo on 4/23/19.
//

#ifndef PARALLEL_FSP_FINITESTATESUBSETHELPERS_H
#define PARALLEL_FSP_FINITESTATESUBSETHELPERS_H

#include "FiniteStateSubset.h"
#include <algorithm>
#include <string>
/*
 * Helper functions for option parsing
 */


namespace cme::parallel{

    std::string part2str( PartitioningType part );

    PartitioningType str2part( std::string str );

    std::string partapproach2str( PartitioningApproach part_approach );

    PartitioningApproach str2partapproach( std::string str );
}
#endif //PARALLEL_FSP_FINITESTATESUBSETHELPERS_H
