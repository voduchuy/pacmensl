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
                return std::string( "Hypergraph" );
            case Hierarchical:
                return std::string( "Hiearchical");
            default:
                return std::string( "Block" );
        }
    }

    PartitioningType str2part( std::string str ) {
        std::transform(str.begin(), str.end(), str.begin(), ::tolower);
        if ( str == "graph") {
            return Graph;
        } else if ( str == "hypergraph") {
            return HyperGraph;
        } else if ( str == "hier" || str == "hierarchical"){
            return Hierarchical;
        }
        else {
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
        std::transform(str.begin(), str.end(), str.begin(), ::tolower);
        if ( str == "from_scratch" || str == "partition" ) {
            return FromScratch;
        } else if ( str == "repart" || str == "repartition") {
            return Repartition;
        } else {
            return Refine;
        }
    }
}
