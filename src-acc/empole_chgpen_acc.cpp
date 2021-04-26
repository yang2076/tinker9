#include "add.h"
#include "empole_chgpen.h"
#include "glob.nblist.h"
#include "image.h"
#include "md.h"
#include "pmestuf.h"
#include "seq_pair_mpole_chgpen.h"
#include "seq_switch.h"
#include "switch.h"
#include "tool/gpu_card.h"

namespace tinker {
#define DEVICE_PTRS                                                            \
   x, y, z, demx, demy, demz, rpole, pcore, pval, palpha, pot, nem, em,        \
      vir_em, trqx, trqy, trqz
template <class Ver, class ETYP, bool CFLX>
void empole_chgpen_acc1()
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_a = Ver::a;
   constexpr bool do_g = Ver::g;
   constexpr bool do_v = Ver::v;

   const real f = electric / dielec;

   real aewald = 0;
   real off;
   if CONSTEXPR (eq<ETYP, EWALD>()) {
      off = switch_off(switch_ewald);
      const PMEUnit pu = epme_unit;
      aewald = pu->aewald;
   } else {
      off = switch_off(switch_mpole);
   }

   const real off2 = off * off;
   const int maxnlst = mlist_unit->maxnlst;
   const auto* mlst = mlist_unit.deviceptr();

   size_t bufsize = buffer_size();
   PairMPoleGrad pgrad;

   MAYBE_UNUSED int GRID_DIM = get_grid_size(BLOCK_DIM);
   #pragma acc parallel async num_gangs(GRID_DIM) vector_length(BLOCK_DIM)\
               present(lvec1,lvec2,lvec3,recipa,recipb,recipc)\
               deviceptr(DEVICE_PTRS,mlst)
   #pragma acc loop gang independent
   for (int i = 0; i < n; ++i) {
      real xi = x[i];
      real yi = y[i];
      real zi = z[i];
      real ci = rpole[i][mpl_pme_0];
      real dix = rpole[i][mpl_pme_x];
      real diy = rpole[i][mpl_pme_y];
      real diz = rpole[i][mpl_pme_z];
      real qixx = rpole[i][mpl_pme_xx];
      real qixy = rpole[i][mpl_pme_xy];
      real qixz = rpole[i][mpl_pme_xz];
      real qiyy = rpole[i][mpl_pme_yy];
      real qiyz = rpole[i][mpl_pme_yz];
      real qizz = rpole[i][mpl_pme_zz];
      real corei = pcore[i];
      real alphai = palpha[i];
      real vali = pval[i];
      MAYBE_UNUSED real gxi = 0, gyi = 0, gzi = 0;
      MAYBE_UNUSED real txi = 0, tyi = 0, tzi = 0;
      MAYBE_UNUSED real poti = 0;

      int nmlsti = mlst->nlst[i];
      int base = i * maxnlst;
      #pragma acc loop vector independent private(pgrad)\
                  reduction(+:gxi,gyi,gzi,txi,tyi,tzi,poti)
      for (int kk = 0; kk < nmlsti; ++kk) {
         int offset = (kk + i * n) & (bufsize - 1);
         int k = mlst->lst[base + kk];
         real xr = x[k] - xi;
         real yr = y[k] - yi;
         real zr = z[k] - zi;
         real corek = pcore[k];
         real alphak = palpha[k];
         real valk = pval[k];

         real r2 = image2(xr, yr, zr);
         if (r2 <= off2) {
            MAYBE_UNUSED real e;
            MAYBE_UNUSED real pota, potb;
            pair_mpole_chgpen<do_e, do_g, ETYP, CFLX>(
               r2, xr, yr, zr, 1,                      //
               ci, dix, diy, diz, corei, vali, alphai, //
               qixx, qixy, qixz, qiyy, qiyz, qizz,     //
               rpole[k][mpl_pme_0], rpole[k][mpl_pme_x], rpole[k][mpl_pme_y],
               rpole[k][mpl_pme_z], corek, valk, alphak, rpole[k][mpl_pme_xx],
               rpole[k][mpl_pme_xy], rpole[k][mpl_pme_xz], rpole[k][mpl_pme_yy],
               rpole[k][mpl_pme_yz], rpole[k][mpl_pme_zz], f, aewald, e, pota,
               potb, pgrad, pentyp);

            if CONSTEXPR (do_a)
               if (e != 0)
                  atomic_add(1, nem, offset);
            if CONSTEXPR (do_e)
               atomic_add(e, em, offset);
            if CONSTEXPR (do_g) {
               gxi += pgrad.frcx;
               gyi += pgrad.frcy;
               gzi += pgrad.frcz;
               atomic_add(-pgrad.frcx, demx, k);
               atomic_add(-pgrad.frcy, demy, k);
               atomic_add(-pgrad.frcz, demz, k);

               txi += pgrad.ttmi[0];
               tyi += pgrad.ttmi[1];
               tzi += pgrad.ttmi[2];
               atomic_add(pgrad.ttmk[0], trqx, k);
               atomic_add(pgrad.ttmk[1], trqy, k);
               atomic_add(pgrad.ttmk[2], trqz, k);

               // virial

               if CONSTEXPR (do_v) {
                  real vxx = -xr * pgrad.frcx;
                  real vxy = -0.5f * (yr * pgrad.frcx + xr * pgrad.frcy);
                  real vxz = -0.5f * (zr * pgrad.frcx + xr * pgrad.frcz);
                  real vyy = -yr * pgrad.frcy;
                  real vyz = -0.5f * (zr * pgrad.frcy + yr * pgrad.frcz);
                  real vzz = -zr * pgrad.frcz;

                  atomic_add(vxx, vxy, vxz, vyy, vyz, vzz, vir_em, offset);
               } // end if (do_v)

               // Charge flux term
               if CONSTEXPR (CFLX) {
                  poti += pota;
                  atomic_add(potb, pot, k);
               }
            } // end if (do_g)
         }    // end if (r2 <= off2)
      }       // end for (int kk)

      if CONSTEXPR (do_g) {
         atomic_add(gxi, demx, i);
         atomic_add(gyi, demy, i);
         atomic_add(gzi, demz, i);
         atomic_add(txi, trqx, i);
         atomic_add(tyi, trqy, i);
         atomic_add(tzi, trqz, i);
         if CONSTEXPR (CFLX) {
            atomic_add(poti, pot, i);
         }
      }

      if CONSTEXPR (eq<ETYP, EWALD>()) {
         const real aewald_sq_2 = 2 * aewald * aewald;
         const real fterm = -f * aewald * 0.5f * (real)(M_2_SQRTPI);
         real cii = ci * ci;
         real dii = dix * dix + diy * diy + diz * diz;
         real qii = 2 * (qixy * qixy + qixz * qixz + qiyz * qiyz) +
            qixx * qixx + qiyy * qiyy + qizz * qizz;

         if CONSTEXPR (do_e) {
            int offset = i & (bufsize - 1);
            real e = fterm *
               (cii +
                aewald_sq_2 * (dii / 3 + 2 * aewald_sq_2 * qii * (real)0.2));
            atomic_add(e, em, offset);
            if CONSTEXPR (do_a)
               atomic_add(1, nem, offset);
         } // end if (do_e)

         if CONSTEXPR (CFLX) {
            real cfl_term = 2 * fterm * ci;
            atomic_add(cfl_term, pot, i);
         }
      }
   } // end for (int i)


   #pragma acc parallel async\
               present(lvec1,lvec2,lvec3,recipa,recipb,recipc)\
               deviceptr(DEVICE_PTRS,mdwexclude,mdwexclude_scale)
   #pragma acc loop independent private(pgrad)
   for (int ii = 0; ii < nmdwexclude; ++ii) {
      int offset = ii & (bufsize - 1);

      int i = mdwexclude[ii][0];
      int k = mdwexclude[ii][1];
      real mscale = mdwexclude_scale[ii][0] - 1;

      real xi = x[i];
      real yi = y[i];
      real zi = z[i];
      real ci = rpole[i][mpl_pme_0];
      real dix = rpole[i][mpl_pme_x];
      real diy = rpole[i][mpl_pme_y];
      real diz = rpole[i][mpl_pme_z];
      real qixx = rpole[i][mpl_pme_xx];
      real qixy = rpole[i][mpl_pme_xy];
      real qixz = rpole[i][mpl_pme_xz];
      real qiyy = rpole[i][mpl_pme_yy];
      real qiyz = rpole[i][mpl_pme_yz];
      real qizz = rpole[i][mpl_pme_zz];
      real corei = pcore[i];
      real alphai = palpha[i];
      real vali = pval[i];

      real xr = x[k] - xi;
      real yr = y[k] - yi;
      real zr = z[k] - zi;

      real corek = pcore[k];
      real alphak = palpha[k];
      real valk = pval[k];

      real r2 = image2(xr, yr, zr);
      if (r2 <= off2 and mscale != 0) {

         MAYBE_UNUSED real e;
         MAYBE_UNUSED real pota, potb;

         // Compute scaled interaction
         pair_mpole_chgpen<do_e, do_g, NON_EWALD, CFLX>(
            r2, xr, yr, zr, mscale,                 //
            ci, dix, diy, diz, corei, vali, alphai, //
            qixx, qixy, qixz, qiyy, qiyz, qizz,     //
            rpole[k][mpl_pme_0], rpole[k][mpl_pme_x], rpole[k][mpl_pme_y],
            rpole[k][mpl_pme_z], corek, valk, alphak, rpole[k][mpl_pme_xx],
            rpole[k][mpl_pme_xy], rpole[k][mpl_pme_xz], rpole[k][mpl_pme_yy],
            rpole[k][mpl_pme_yz], rpole[k][mpl_pme_zz], f, 0, e, pota, potb,
            pgrad, pentyp);

         if CONSTEXPR (do_a)
            if (mscale == -1 and e != 0)
               atomic_add(-1, nem, offset);

         if CONSTEXPR (do_e)
            atomic_add(e, em, offset);
         if CONSTEXPR (do_g) {
            atomic_add(pgrad.frcx, demx, i);
            atomic_add(pgrad.frcy, demy, i);
            atomic_add(pgrad.frcz, demz, i);
            atomic_add(-pgrad.frcx, demx, k);
            atomic_add(-pgrad.frcy, demy, k);
            atomic_add(-pgrad.frcz, demz, k);

            atomic_add(pgrad.ttmi[0], trqx, i);
            atomic_add(pgrad.ttmi[1], trqy, i);
            atomic_add(pgrad.ttmi[2], trqz, i);
            atomic_add(pgrad.ttmk[0], trqx, k);
            atomic_add(pgrad.ttmk[1], trqy, k);
            atomic_add(pgrad.ttmk[2], trqz, k);

            // virial

            if CONSTEXPR (do_v) {
               real vxx = -xr * pgrad.frcx;
               real vxy = -0.5f * (yr * pgrad.frcx + xr * pgrad.frcy);
               real vxz = -0.5f * (zr * pgrad.frcx + xr * pgrad.frcz);
               real vyy = -yr * pgrad.frcy;
               real vyz = -0.5f * (zr * pgrad.frcy + yr * pgrad.frcz);
               real vzz = -zr * pgrad.frcz;

               atomic_add(vxx, vxy, vxz, vyy, vyz, vzz, vir_em, offset);
            } // end if (do_v)

            if CONSTEXPR (CFLX) {
               atomic_add(pota, pot, i);
               atomic_add(potb, pot, k);
            }
         } // end if (do_g)
      }    // end if (r2 <= off2)
   }       // end for (int ii)
}


template <class Ver, int CFLX>
void empole_generic_ewald_recip_acc()
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_g = Ver::g;
   constexpr bool do_v = Ver::v;

   auto bufsize = buffer_size();

   const PMEUnit pu = epme_unit;
   cmp_to_fmp(pu, cmp, fmp);
   grid_mpole(pu, fmp);
   fftfront(pu);
   if CONSTEXPR (do_v) {
      if (vir_m) {
         pme_conv(pu, vir_m);
         auto size = buffer_size() * virial_buffer_traits::value;
         #pragma acc parallel loop independent async deviceptr(vir_m,vir_em)
         for (size_t i = 0; i < size; ++i) {
            vir_em[0][i] += vir_m[0][i];
         }
      } else {
         pme_conv(pu, vir_em);
      }
   } else {
      pme_conv(pu);
   }
   fftback(pu);
   fphi_mpole(pu);
   fphi_to_cphi(pu, fphi, cphi);

   auto& st = *pu;
   const int nfft1 = st.nfft1;
   const int nfft2 = st.nfft2;
   const int nfft3 = st.nfft3;
   const real f = electric / dielec;

   #pragma acc parallel loop independent async\
               present(lvec1,lvec2,lvec3,recipa,recipb,recipc)\
               deviceptr(demx,demy,demz,\
               cmp,fmp,cphi,fphi,em,vir_em,trqx,trqy,trqz,pot)
   for (int i = 0; i < n; ++i) {
      constexpr int deriv1[] = {2, 5, 8, 9, 11, 16, 18, 14, 15, 20};
      constexpr int deriv2[] = {3, 8, 6, 10, 14, 12, 19, 16, 20, 17};
      constexpr int deriv3[] = {4, 9, 10, 7, 15, 17, 13, 20, 18, 19};

      int offset = i & (bufsize - 1);
      real e = 0;
      real f1 = 0;
      real f2 = 0;
      real f3 = 0;

      #pragma acc loop seq
      for (int k = 0; k < 10; ++k) {
         if CONSTEXPR (do_e)
            e += fmp[i][k] * fphi[i][k];
         if CONSTEXPR (do_g) {
            f1 += fmp[i][k] * fphi[i][deriv1[k] - 1];
            f2 += fmp[i][k] * fphi[i][deriv2[k] - 1];
            f3 += fmp[i][k] * fphi[i][deriv3[k] - 1];
         }
      } // end for (int k)

      // increment the permanent multipole energy and gradient

      if CONSTEXPR (do_e)
         atomic_add(0.5f * e * f, em, offset);

      if CONSTEXPR (do_g) {
         f1 *= nfft1;
         f2 *= nfft2;
         f3 *= nfft3;

         real h1 = recipa.x * f1 + recipb.x * f2 + recipc.x * f3;
         real h2 = recipa.y * f1 + recipb.y * f2 + recipc.y * f3;
         real h3 = recipa.z * f1 + recipb.z * f2 + recipc.z * f3;

         atomic_add(h1 * f, demx, i);
         atomic_add(h2 * f, demy, i);
         atomic_add(h3 * f, demz, i);

         // resolve site torques then increment forces and virial

         real tem1 = cmp[i][3] * cphi[i][2] - cmp[i][2] * cphi[i][3] +
            2 * (cmp[i][6] - cmp[i][5]) * cphi[i][9] + cmp[i][8] * cphi[i][7] +
            cmp[i][9] * cphi[i][5] - cmp[i][7] * cphi[i][8] -
            cmp[i][9] * cphi[i][6];
         real tem2 = cmp[i][1] * cphi[i][3] - cmp[i][3] * cphi[i][1] +
            2 * (cmp[i][4] - cmp[i][6]) * cphi[i][8] + cmp[i][7] * cphi[i][9] +
            cmp[i][8] * cphi[i][6] - cmp[i][8] * cphi[i][4] -
            cmp[i][9] * cphi[i][7];
         real tem3 = cmp[i][2] * cphi[i][1] - cmp[i][1] * cphi[i][2] +
            2 * (cmp[i][5] - cmp[i][4]) * cphi[i][7] + cmp[i][7] * cphi[i][4] +
            cmp[i][9] * cphi[i][8] - cmp[i][7] * cphi[i][5] -
            cmp[i][8] * cphi[i][9];
         tem1 *= f;
         tem2 *= f;
         tem3 *= f;

         atomic_add(tem1, trqx, i);
         atomic_add(tem2, trqy, i);
         atomic_add(tem3, trqz, i);

         if CONSTEXPR (do_v) {
            real vxx = -cmp[i][1] * cphi[i][1] - 2 * cmp[i][4] * cphi[i][4] -
               cmp[i][7] * cphi[i][7] - cmp[i][8] * cphi[i][8];
            real vxy =
               -0.5f * (cmp[i][2] * cphi[i][1] + cmp[i][1] * cphi[i][2]) -
               (cmp[i][4] + cmp[i][5]) * cphi[i][7] -
               0.5f * cmp[i][7] * (cphi[i][4] + cphi[i][5]) -
               0.5f * (cmp[i][8] * cphi[i][9] + cmp[i][9] * cphi[i][8]);
            real vxz =
               -0.5f * (cmp[i][3] * cphi[i][1] + cmp[i][1] * cphi[i][3]) -
               (cmp[i][4] + cmp[i][6]) * cphi[i][8] -
               0.5f * cmp[i][8] * (cphi[i][4] + cphi[i][6]) -
               0.5f * (cmp[i][7] * cphi[i][9] + cmp[i][9] * cphi[i][7]);
            real vyy = -cmp[i][2] * cphi[i][2] - 2 * cmp[i][5] * cphi[i][5] -
               cmp[i][7] * cphi[i][7] - cmp[i][9] * cphi[i][9];
            real vyz =
               -0.5f * (cmp[i][3] * cphi[i][2] + cmp[i][2] * cphi[i][3]) -
               (cmp[i][5] + cmp[i][6]) * cphi[i][9] -
               0.5f * cmp[i][9] * (cphi[i][5] + cphi[i][6]) -
               0.5f * (cmp[i][7] * cphi[i][8] + cmp[i][8] * cphi[i][7]);
            real vzz = -cmp[i][3] * cphi[i][3] - 2 * cmp[i][6] * cphi[i][6] -
               cmp[i][8] * cphi[i][8] - cmp[i][9] * cphi[i][9];
            vxx *= f;
            vxy *= f;
            vxz *= f;
            vyy *= f;
            vyz *= f;
            vzz *= f;

            atomic_add(vxx, vxy, vxz, vyy, vyz, vzz, vir_em, offset);
         } // end if (do_v)
         if CONSTEXPR (CFLX) {
            atomic_add(f * cphi[i][0], pot, i);
         }
      } // end if (do_g)
   }    // end for (int i)
}


void empole_chgpen_nonewald_acc(int vers, int use_cf)
{

   if (use_cf) {
      if (vers == calc::v0) {
         // empole_chgpen_acc1<calc::V0, NON_EWALD, 1>();
         assert(false && "CFLX must compute gradient.");
      } else if (vers == calc::v1) {
         empole_chgpen_acc1<calc::V1, NON_EWALD, 1>();
      } else if (vers == calc::v3) {
         // empole_chgpen_acc1<calc::V3, NON_EWALD, 1>();
         assert(false && "CFLX must compute gradient.");
      } else if (vers == calc::v4) {
         empole_chgpen_acc1<calc::V4, NON_EWALD, 1>();
      } else if (vers == calc::v5) {
         empole_chgpen_acc1<calc::V5, NON_EWALD, 1>();
      } else if (vers == calc::v6) {
         empole_chgpen_acc1<calc::V6, NON_EWALD, 1>();
      }
   } else {
      if (vers == calc::v0) {
         empole_chgpen_acc1<calc::V0, NON_EWALD, 0>();
      } else if (vers == calc::v1) {
         empole_chgpen_acc1<calc::V1, NON_EWALD, 0>();
      } else if (vers == calc::v3) {
         empole_chgpen_acc1<calc::V3, NON_EWALD, 0>();
      } else if (vers == calc::v4) {
         empole_chgpen_acc1<calc::V4, NON_EWALD, 0>();
      } else if (vers == calc::v5) {
         empole_chgpen_acc1<calc::V5, NON_EWALD, 0>();
      } else if (vers == calc::v6) {
         empole_chgpen_acc1<calc::V6, NON_EWALD, 0>();
      }
   }
}


void empole_chgpen_ewald_real_self_acc(int vers, int use_cf)
{
   if (use_cf) {
      if (vers == calc::v0) {
         // empole_chgpen_acc1<calc::V0, EWALD, 1>();
         assert(false && "CFLX must compute gradient.");
      } else if (vers == calc::v1) {
         empole_chgpen_acc1<calc::V1, EWALD, 1>();
      } else if (vers == calc::v3) {
         // empole_chgpen_acc1<calc::V3, EWALD, 1>();
         assert(false && "CFLX must compute gradient.");
      } else if (vers == calc::v4) {
         empole_chgpen_acc1<calc::V4, EWALD, 1>();
      } else if (vers == calc::v5) {
         empole_chgpen_acc1<calc::V5, EWALD, 1>();
      } else if (vers == calc::v6) {
         empole_chgpen_acc1<calc::V6, EWALD, 1>();
      }
   } else {
      if (vers == calc::v0) {
         empole_chgpen_acc1<calc::V0, EWALD, 0>();
      } else if (vers == calc::v1) {
         empole_chgpen_acc1<calc::V1, EWALD, 0>();
      } else if (vers == calc::v3) {
         empole_chgpen_acc1<calc::V3, EWALD, 0>();
      } else if (vers == calc::v4) {
         empole_chgpen_acc1<calc::V4, EWALD, 0>();
      } else if (vers == calc::v5) {
         empole_chgpen_acc1<calc::V5, EWALD, 0>();
      } else if (vers == calc::v6) {
         empole_chgpen_acc1<calc::V6, EWALD, 0>();
      }
   }
}

void empole_chgpen_ewald_recip_acc(int vers, int use_cf)
{
   if (use_cf) {
      if (vers == calc::v0)
         // empole_generic_ewald_recip_acc<calc::V0, 1>();
         assert(false && "CFLX must compute gradient.");
      else if (vers == calc::v1)
         empole_generic_ewald_recip_acc<calc::V1, 1>();
      else if (vers == calc::v3)
         // empole_generic_ewald_recip_acc<calc::V3, 1>();
         assert(false && "CFLX must compute gradient.");
      else if (vers == calc::v4)
         empole_generic_ewald_recip_acc<calc::V4, 1>();
      else if (vers == calc::v5)
         empole_generic_ewald_recip_acc<calc::V5, 1>();
      else if (vers == calc::v6)
         empole_generic_ewald_recip_acc<calc::V6, 1>();
   } else {
      if (vers == calc::v0)
         empole_generic_ewald_recip_acc<calc::V0, 0>();
      else if (vers == calc::v1)
         empole_generic_ewald_recip_acc<calc::V1, 0>();
      else if (vers == calc::v3)
         empole_generic_ewald_recip_acc<calc::V3, 0>();
      else if (vers == calc::v4)
         empole_generic_ewald_recip_acc<calc::V4, 0>();
      else if (vers == calc::v5)
         empole_generic_ewald_recip_acc<calc::V5, 0>();
      else if (vers == calc::v6)
         empole_generic_ewald_recip_acc<calc::V6, 0>();
   }
}
}
