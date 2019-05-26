//
// Created by Huy Vo on 4/23/19.
//

/*
 * Helper functions for option parsing
 */
#include "FiniteStateSubsetHelpers.h"

namespace cme::parallel{
    std::string part2str( PartitioningType part ) {
        switch ( part ) {
            case Graph:
                return std::string( "Graph" );
            case HyperGraph:
                return std::string( "Hyper-graph" );
            default:
                return std::string( "Block" );
        }
    }

    PartitioningType str2part( std::string str ) {
        if ( str == "graph" || str == "Graph" || str == "GRAPH" ) {
            return Graph;
        } else if ( str == "hypergraph" || str == "HyperGraph" || str == "HYPERGRAPH" ) {
            return HyperGraph;
        } else {
            return Block;
        }
    }

    std::string partapproach2str( PartitioningApproach part_approach ) {
        switch ( part_approach ) {
            case FromScratch:
                return std::string( "fromscratch" );
            case Repartition:
                return std::string( "repartition" );
            default:
                return std::string( "refine" );
        }
    }

    PartitioningApproach str2partapproach( std::string str ) {
        if ( str == "from_scratch" || str == "partition" || str == "FromScratch" || str == "FROMSCRATCH" ) {
            return FromScratch;
        } else if ( str == "repart" || str == "repartition" || str == "REPARTITION" || str == "Repart" ||
                    str == "Repartition" ) {
            return Repartition;
        } else {
            return Refine;
        }
    }
}
