
#include <util/cme_util.h>

#include "cme_util.h"

namespace cme {
    namespace parallel {

    }

    int ParaFSP_init( int *argc, char ***argv, const char *help ) {
        PetscErrorCode ierr;
        MPI_Init( argc, argv );
        ierr = PetscInitialize( argc, argv, ( char * ) 0, help );
        CHKERRQ( ierr );
        float ver;
        ierr = Zoltan_Initialize( *argc, *argv, &ver );
        CHKERRQ( ierr );
        return 0;
    }

    int ParaFSP_finalize( ) {
        PetscErrorCode ierr;
        ierr = PetscFinalize( );
        CHKERRQ( ierr );
        MPI_Finalize( );
        return ierr;
        return 0;
    }

    void sequential_action( MPI_Comm comm, std::function< void( void * ) > action, void *data ) {
        int my_rank, comm_size;
        MPI_Comm_rank( comm, &my_rank );
        MPI_Comm_size( comm, &comm_size );

        if ( comm_size == 1 ) {
            action( data );
            return;
        }

        int print;
        MPI_Status status;
        if ( my_rank == 0 ) {
            std::cout << "Processor " << my_rank << "\n";
            action( data );
            MPI_Send( &print, 1, MPI_INT, my_rank + 1, 1, comm );
        } else {
            MPI_Recv( &print, 1, MPI_INT, my_rank - 1, 1, comm, &status );
            std::cout << "Processor " << my_rank << "\n";
            action( data );
            if ( my_rank < comm_size - 1 ) {
                MPI_Send( &print, 1, MPI_INT, my_rank + 1, 1, comm );
            }
        }
        MPI_Barrier( comm );
    }

    double round2digit( double x ) {
        if (x == 0.0e0) return x;
        double p1 = std::pow(10.0e0, round(log10(x) - SQR1) - 1.0e0);
        return trunc(x/p1 + 0.55e0)*p1;
    }
}
