#ifndef TINKER_GPU_E_STRBND_H_
#define TINKER_GPU_E_STRBND_H_

#include "util_cxx.h"
#include "util_rc_man.h"

TINKER_NAMESPACE_BEGIN
extern int nstrbnd;
extern int (*isb)[3];
extern real (*sbk)[2];
extern real stbnunit;

extern real* eba;
extern real* vir_eba;

void estrbnd_data(rc_t rc);

void estrbnd(int vers);
TINKER_NAMESPACE_END

#endif
