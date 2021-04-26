#pragma once
#include "mathfunc.h"
#include "seq_def.h"


namespace tinker {
#pragma acc routine seq
template <int ORDER>
SEQ_CUDA
inline void damp_gordon2(real* restrict dmpik, real* restrict dmpi,
                         real* restrict dmpk, real r, real alphai, real alphak)
{
#if TINKER_REAL_SIZE == 8
   real eps = 0.001f;
#elif TINKER_REAL_SIZE == 4
   real eps = 0.05f;
#endif

   real diff = REAL_ABS(alphai - alphak);

   if (diff < eps)
      alphai = 0.5f * (alphai + alphak);

   real dampi = alphai * r;
   real dampk = alphak * r;
   real expi = REAL_EXP(-dampi);
   real expk = REAL_EXP(-dampk);

   real dampi2 = dampi * dampi;
   real dampi3 = dampi * dampi2;

   // divisions
   const real div3 = 1 / ((real)3);
   const real div15 = 1 / ((real)15);

   // GORDON2
   // core-valence
   dmpi[0] = expi;
   dmpi[1] = (1 + dampi) * expi;
   dmpi[2] = (1 + dampi + dampi2 * div3) * expi;
   dmpi[3] = (1 + dampi + 0.4f * dampi2 + dampi3 * div15) * expi;

   if (diff < eps) {
      dmpk[0] = dmpi[0];
      dmpk[1] = dmpi[1];
      dmpk[2] = dmpi[2];
      dmpk[3] = dmpi[3];

      // valence-valence
      real dampi4 = dampi2 * dampi2;
      real dampi5 = dampi2 * dampi3;
      const real div6 = 1 / ((real)6);
      const real div30 = 1 / ((real)30);
      const real div105 = 1 / ((real)105);
      const real div210 = 1 / ((real)210);

      dmpik[0] = (1.0f + 0.5f * dampi) * expi;
      dmpik[1] = (1.0f + dampi + 0.5f * dampi2) * expi;
      dmpik[2] = (1.0f + dampi + 0.5f * dampi2 + dampi3 * div6) * expi;
		  dmpik[3] = (1.0f + dampi + 0.5f * dampi2 + dampi3 * div6 + dampi4 * div30) * expi;
      dmpik[4] = (1.0f + dampi + 0.5f * dampi2 + dampi3 * div6 + 4.0f * dampi4 * div105 + dampi5*div210) * expi;
      if CONSTEXPR (ORDER > 9) {
         real dampi6 = dampi3 * dampi3;
         const real div126 = 1 / ((real)126);
         const real div315 = 1 / ((real)315);
         const real div1890 = 1 / ((real)1890);
         dmpik[5] = (1.0f + dampi + 0.5f * dampi2 + dampi3 * div6 + 5.0f * dampi4 * div126 
				 					+  2.0f * dampi5 * div315 + dampi6 * div1890) * expi;
      }
   } else {
      real dampi2 = dampi * dampi;
      real dampi3 = dampi * dampi2;
      real dampi4 = dampi2 * dampi2;
      real dampk2 = dampk * dampk;
      real dampk3 = dampk * dampk2;
      real dampk4 = dampk2 * dampk2;

      const real div7 = 1 / ((real)7);
      const real div21 = 1 / ((real)21);
      const real div105 = 1 / ((real)105);
   		dmpk[0] = expk;
   		dmpk[1] = (1 + dampk) * expk;
   		dmpk[2] = (1 + dampk + dampk2 * div3) * expk;
   		dmpk[3] = (1 + dampk + 0.4f * dampk2 + dampk3 * div15) * expk;

      // valence-valence
      real alphai2 = alphai * alphai;
      real alphak2 = alphak * alphak;
      real alphaik = ((alphak + alphai) * (alphak - alphai));
      real termi = alphak2 / alphaik;
      real termk = -alphai2 / alphaik;

      dmpik[0] = termi * expi + termk * expk;
      dmpik[1] = termi * (1 + dampi) * expi 
							 + termk * (1 + dampk) * expk;
      dmpik[2] = termi * (1 + dampi + dampi2 * div3) * expi 
							 + termk * (1 + dampk + dampk2 * div3) * expk;
      dmpik[3] = termi * (1 + dampi + 0.4f * dampi2 + dampi3 * div15)*expi 
							 + termk * (1 + dampk + 0.4f * dampk2 + dampk3 * div15)*expk;
      dmpik[4] = termi * (1 + dampi + 3.0f * dampi2 * div7 + 2.0f * dampi3 * div21 + dampi4 * div105) * expi 
							 + termk * (1 + dampk + 3.0f * dampk2 * div7 + 2.0f * dampk3 * div21 + dampk4 * div105) * expk;     

      if CONSTEXPR (ORDER > 9) {
         const real div9 = 1 / ((real)9);
         const real div63 = 1 / ((real)63);
         const real div945 = 1 / ((real)945);
      	 real dampi5 = dampi2 * dampi3;
      	 real dampk5 = dampk2 * dampk3;
         dmpik[5] = termi * (1 + dampi + 4.0f * dampi2 * div9 + dampi3 * div9 + dampi4 * div63 + dampi5 * div945) * expi
                  + termk * (1 + dampk + 4.0f * dampk2 * div9 + dampk3 * div9 + dampk4 * div63 + dampk5 * div945) * expk;
      }
   }
}
}
