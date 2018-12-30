
#include <util/cme_util.h>

#include "cme_util.h"

namespace cme {
    namespace parallel {

    }

    int ParaFSP_init(int *argc, char ***argv, const char *help) {
        PetscErrorCode ierr;
        MPI_Init(argc, argv);
        ierr = PetscInitialize(argc, argv, (char *) 0, help);
        CHKERRQ(ierr);
        float ver;
        ierr = Zoltan_Initialize(*argc, *argv, &ver);
        CHKERRQ(ierr);
        return 0;
    }

    int ParaFSP_finalize() {
        PetscErrorCode ierr;
        ierr = PetscFinalize();
        CHKERRQ(ierr);
        MPI_Finalize();
        return ierr;
        return 0;
    }
}
