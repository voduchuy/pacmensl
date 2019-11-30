#include <cvode/cvode.h>
#include <cvode/cvode_spils.h>

#ifdef __cpp  /* wrapper to enable C++ usage */
extern "C" {
#endif

typedef realtype (*EPICNumJacDelta)(N_Vector u, N_Vector v, realtype normu, realtype normv, void *user_data);

#ifdef __cpp
}
#endif