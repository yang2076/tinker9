#pragma once
#include "mathfunc.h"
#include "seq_def.h"
#include "seq_switch.h"
#include "tool/io_print.h"

namespace tinker {
#pragma acc routine seq
template <bool DO_G>
SEQ_CUDA
void pair_chgtrn(real r, real cut, real off, real mscale, real f, real alphai,
                 real chgi, real alphak, real chgk, e_prec& restrict e,
                 e_prec& restrict de, chgtrn_t ctrntyp)
{
   f *= mscale;
   real expi = 0.;
   real expk = 0.;
   real alphaik = 0.;
   real chgik = 0.;

   if (ctrntyp == chgtrn_t::SEPARATE)
   {
      real expi = REAL_EXP(-alphai * r);
      real expk = REAL_EXP(-alphak * r);
      e = -chgi * expk - chgk * expi;
   }
   else if (ctrntyp == chgtrn_t::COMBINED)
   {
      alphaik = 0.5 * (alphai + alphak); 
      chgik = REAL_SQRT(chgi * chgk);
      e = -chgik * REAL_EXP(-alphaik * r);
   }
   e *= f;
   //printf("%10.5f %10.5f %10.5f %10.5f\n",alphaik, chgik, r, e);

   if CONSTEXPR (DO_G) {
      if (ctrntyp == chgtrn_t::SEPARATE) {
         de = chgi * expk * alphak + chgk * expi * alphai;
         de *= f;
      }
      else if (ctrntyp == chgtrn_t::COMBINED) {
         de = alphaik * chgik * REAL_EXP(-alphaik * r);
		 de *= f;
      }
   }

   if (r > cut) {
      real taper, dtaper;
      switch_taper5<DO_G>(r, cut, off, taper, dtaper);
      if CONSTEXPR (DO_G)
         de = e * dtaper + de * taper;
      e *= taper;
   }
}
}
