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

#ifndef PACMENSL_SRC_SYS_ERRORHANDLING_H_
#define PACMENSL_SRC_SYS_ERRORHANDLING_H_
#include <cstring>

using PacmenslErrorCode = int;


#define PACMENSLCHKERRQ(ierr){\
    if (ierr != 0) {\
        int rank;\
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);\
        printf("PACMENSL Error: function %s line %d file %s on rank %d \n.", __func__, __LINE__, __FILE__, rank);\
        if (!std::strcmp(__func__, "main")) {\
          MPI_Abort(MPI_COMM_WORLD, ierr);\
        }\
        else{\
          return ierr;\
        }\
    }\
}

#define PACMENSLCHKERRTHROW(ierr){\
if (ierr != 0) {\
        int rank;\
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);\
        std::ostringstream msg;\
        msg << "PACMENSL Error: Line " << __LINE__ << " in " << __FILE__ << " funcion " << __func__ << "rank " << rank << ".";\
        throw std::runtime_error(msg.str());\
    }\
}

#define ZOLTANCHKERRABORT(comm, ierr){\
            if (ierr == ZOLTAN_FATAL){\
                PetscPrintf(comm, "Zoltan Fatal in %s at line %d\n", __FILE__, __LINE__);\
                MPI_Abort(comm, -1);\
            }\
}

#define ZOLTANCHKERRQ(ierr){\
if (ierr != ZOLTAN_OK && ierr != ZOLTAN_WARN) {\
        int rank;\
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);\
        if (ierr == ZOLTAN_FATAL)\
        {\
          printf("Call to Zoltan returns fatal error: function %s line %d file %s on rank %d \n.", __func__, __LINE__, __FILE__, rank);\
          }\
        else{\
          printf("Call to Zoltan returns memory error: function %s line %d file %s on rank %d \n.", __func__, __LINE__, __FILE__, rank);\
        }\
        if (!std::strcmp(__func__, "main")) MPI_Abort(MPI_COMM_WORLD, ierr);\
        return ierr;\
    }\
}\

#define MPICHKERRABORT(comm, ierr){\
            if (ierr != MPI_SUCCESS){\
                PetscPrintf(comm, "MPI error in %s at line %d\n", __FILE__, __LINE__);\
                MPI_Abort(comm, -1);\
            }\
}

#define MPICHKERRTHROW(ierr){\
if (ierr != MPI_SUCCESS) {\
        int rank;\
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);\
        std::ostringstream msg;\
        msg << "Call to MPI function returns error code " << ierr <<" on line " << __LINE__ << " in " << __FILE__ << " funcion " << __func__ << "rank " << rank << ".";\
        throw std::runtime_error(msg.str());\
    }\
}

#define PETSCCHKERRTHROW(ierr){\
if (ierr != 0) {\
        int rank;\
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);\
        std::ostringstream msg;\
        msg << "Call to PETSc function returns error code " << ierr <<" on line " << __LINE__ << " in " << __FILE__ << " funcion " << __func__ << "rank " << rank << ".";\
        throw std::runtime_error(msg.str());\
    }\
}


#define CVODECHKERRABORT(comm, flag){\
    if (flag < 0) \
    {\
    PetscPrintf(comm, "\nSUNDIALS_ERROR: function failed in file %s line %d with flag = %d\n\n",\
    __FILE__,__LINE__, flag);\
    MPI_Abort(comm, 1);\
    }\
    }
#define CVODECHKERRQ(flag){\
    if (flag < 0) \
    {\
    int rank;\
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);\
    printf("\nSUNDIALS_ERROR: function failed on rank %d in file %s line %d with flag = %d\n\n",\
    rank,__FILE__,__LINE__, flag);\
    return -1;\
    }\
    }


#endif //PACMENSL_SRC_SYS_ERRORHANDLING_H_


