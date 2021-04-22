#include "add.h"
#include "emplar.h"
#include "empole.h"
#include "empole_self.h"
#include "epolar.h"
#include "epolar_trq.h"
#include "glob.spatial.h"
#include "image.h"
#include "launch.h"
#include "md.h"
#include "pme.h"
#include "seq_damp.h"
#include "seq_triangle.h"
#include "switch.h"
#include "tool/gpu_card.h"


namespace tinker {
// Rt Q = G
__device__
void rotQI2GVector(const real (&restrict rot)[3][3], real3 qif,
                   real3& restrict glf)
{
   glf = make_real3(dot3(rot[0][0], rot[1][0], rot[2][0], qif),
                    dot3(rot[0][1], rot[1][1], rot[2][1], qif),
                    dot3(rot[0][2], rot[1][2], rot[2][2], qif));
}


// R G = Q
__device__
void rotG2QIVector(const real (&restrict rot)[3][3], real3 glf,
                   real3& restrict qif)
{
   qif = make_real3(dot3(rot[0][0], rot[0][1], rot[0][2], glf),
                    dot3(rot[1][0], rot[1][1], rot[1][2], glf),
                    dot3(rot[2][0], rot[2][1], rot[2][2], glf));
}


// R G Rt = Q
__device__
void rotG2QIMat_v1(const real (&restrict rot)[3][3], //
                   real glxx, real glxy, real glxz,  //
                   real glyy, real glyz, real glzz,  //
                   real& restrict qixx, real& restrict qixy,
                   real& restrict qixz, real& restrict qiyy,
                   real& restrict qiyz, real& restrict qizz)
{
   real gl[3][3] = {{glxx, glxy, glxz}, {glxy, glyy, glyz}, {glxz, glyz, glzz}};
   real out[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
   // out[i][j] = sum(k,m) R[i][k] gl[k][m] Rt[m][j]
   //           = sum(k,m) R[i][k] gl[k][m] R[j][m]
   for (int i = 0; i < 2; ++i)
      for (int j = i; j < 3; ++j)
         for (int k = 0; k < 3; ++k)
            for (int m = 0; m < 3; ++m)
               out[i][j] += rot[i][k] * gl[k][m] * rot[j][m];
   qixx = out[0][0];
   qixy = out[0][1];
   qixz = out[0][2];
   qiyy = out[1][1];
   qiyz = out[1][2];
   // qizz = out[2][2];
   qizz = -(out[0][0] + out[1][1]);
}


// R G Rt = Q
__device__
void rotG2QIMat_v2(const real (&restrict r)[3][3],  //
                   real glxx, real glxy, real glxz, //
                   real glyy, real glyz, real glzz, //
                   real& restrict qixx, real& restrict qixy,
                   real& restrict qixz, real& restrict qiyy,
                   real& restrict qiyz, real& restrict qizz)
{
   // clang-format off
   qixx=r[0][0]*(r[0][0]*glxx+2*r[0][1]*glxy) + r[0][1]*(r[0][1]*glyy+2*r[0][2]*glyz) + r[0][2]*(r[0][2]*glzz+2*r[0][0]*glxz);
   qiyy=r[1][0]*(r[1][0]*glxx+2*r[1][1]*glxy) + r[1][1]*(r[1][1]*glyy+2*r[1][2]*glyz) + r[1][2]*(r[1][2]*glzz+2*r[1][0]*glxz);
   qixy=r[0][0]*(r[1][0]*glxx+r[1][1]*glxy+r[1][2]*glxz) + r[0][1]*(r[1][0]*glxy+r[1][1]*glyy+r[1][2]*glyz) + r[0][2]*(r[1][0]*glxz+r[1][1]*glyz+r[1][2]*glzz);
   qixz=r[0][0]*(r[2][0]*glxx+r[2][1]*glxy+r[2][2]*glxz) + r[0][1]*(r[2][0]*glxy+r[2][1]*glyy+r[2][2]*glyz) + r[0][2]*(r[2][0]*glxz+r[2][1]*glyz+r[2][2]*glzz);
   qiyz=r[1][0]*(r[2][0]*glxx+r[2][1]*glxy+r[2][2]*glxz) + r[1][1]*(r[2][0]*glxy+r[2][1]*glyy+r[2][2]*glyz) + r[1][2]*(r[2][0]*glxz+r[2][1]*glyz+r[2][2]*glzz);
   // clang-format on
   qizz = -(qixx + qiyy);
}


#define rotG2QIMatrix rotG2QIMat_v2


template <class Ver, class ETYP>
__device__
void pair_mplar(                                                          //
   real r2, real3 dR, real mscale, real dscale, real pscale, real uscale, //
   real ci, real3 Id, real Iqxx, real Iqxy, real Iqxz, real Iqyy, real Iqyz,
   real Iqzz, real3 Iud, real3 Iup, real pdi, real pti, //
   real ck, real3 Kd, real Kqxx, real Kqxy, real Kqxz, real Kqyy, real Kqyz,
   real Kqzz, real3 Kud, real3 Kup, real pdk, real ptk, //
   real f, real aewald,                                 //
   real& restrict frcxi, real& restrict frcyi, real& restrict frczi,
   real& restrict frcxk, real& restrict frcyk, real& restrict frczk,
   real& restrict trqxi, real& restrict trqyi, real& restrict trqzi,
   real& restrict trqxk, real& restrict trqyk, real& restrict trqzk,
   real& restrict eo, real& restrict voxx, real& restrict voxy,
   real& restrict voxz, real& restrict voyy, real& restrict voyz,
   real& restrict vozz)
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_g = Ver::g;
   constexpr bool do_v = Ver::v;


   // a rotation matrix that rotates (xr,yr,zr) to (0,0,r); R G = Q
   real rot[3][3];
   real bn[6];
   real sr3, sr5, sr7, sr9;
   real r = REAL_SQRT(r2);
   real invr1 = REAL_RECIP(r);
   {
      real rr2 = invr1 * invr1;
      real rr1 = invr1;
      real rr3 = rr1 * rr2;
      real rr5 = 3 * rr3 * rr2;
      real rr7 = 5 * rr5 * rr2;
      real rr9 = 7 * rr7 * rr2;
      real rr11;
      if CONSTEXPR (do_g) {
         rr11 = 9 * rr9 * rr2;
      }


      if CONSTEXPR (eq<ETYP, EWALD>()) {
         if CONSTEXPR (!do_g) {
            damp_ewald<5>(bn, r, invr1, rr2, aewald);
         } else {
            damp_ewald<6>(bn, r, invr1, rr2, aewald);
         }
      } else if CONSTEXPR (eq<ETYP, NON_EWALD>()) {
         bn[0] = rr1;
         bn[1] = rr3;
         bn[2] = rr5;
         bn[3] = rr7;
         bn[4] = rr9;
         if CONSTEXPR (do_g) {
            bn[5] = rr11;
         }
      }


      // if use_thole
      real ex3, ex5, ex7, ex9;
      damp_thole4(r, pdi, pti, pdk, ptk, ex3, ex5, ex7, ex9);
      sr3 = bn[1] - ex3 * rr3;
      sr5 = bn[2] - ex5 * rr5;
      sr7 = bn[3] - ex7 * rr7;
      sr9 = bn[4] - ex9 * rr9;
      // end if use_thole


      real3 rotz = invr1 * dR;
      // pick a random vector as rotx; rotx and rotz cannot be parallel
      real3 rotx = rotz;
      if (dR.y != 0 || dR.z != 0)
         rotx.x += 1;
      else
         rotx.y += 1;
      // Gram–Schmidt process for rotx with respect to rotz
      rotx -= dot3(rotx, rotz) * rotz;
      // normalize rotx
      real invxlen = REAL_RSQRT(dot3(rotx, rotx));
      rotx = invxlen * rotx;
      real3 roty = cross(rotz, rotx);
      rot[0][0] = rotx.x;
      rot[0][1] = rotx.y;
      rot[0][2] = rotx.z;
      rot[1][0] = roty.x;
      rot[1][1] = roty.y;
      rot[1][2] = roty.z;
      rot[2][0] = rotz.x;
      rot[2][1] = rotz.y;
      rot[2][2] = rotz.z;
   }


   real3 di, dk;
   rotG2QIVector(rot, Id, di);
   rotG2QIVector(rot, Kd, dk);
   real qixx, qixy, qixz, qiyy, qiyz, qizz;
   real qkxx, qkxy, qkxz, qkyy, qkyz, qkzz;
   rotG2QIMatrix(rot, Iqxx, Iqxy, Iqxz, Iqyy, Iqyz, Iqzz, qixx, qixy, qixz,
                 qiyy, qiyz, qizz);
   rotG2QIMatrix(rot, Kqxx, Kqxy, Kqxz, Kqyy, Kqyz, Kqzz, qkxx, qkxy, qkxz,
                 qkyy, qkyz, qkzz);
   real3 uid, uip;
   rotG2QIVector(rot, Iud, uid);
   rotG2QIVector(rot, Iup, uip);
   real3 ukd, ukp;
   rotG2QIVector(rot, Kud, ukd);
   rotG2QIVector(rot, Kup, ukp);


   // phi,dphi/d(x,y,z),d2phi/dd(xx,yy,zz,xy,xz,yz)
   //   0        1 2 3            4  5  6  7  8  9
   real phi1[10] = {0};
   real phi2[10] = {0};
   real phi1z[10] = {0};


   if CONSTEXPR (eq<ETYP, EWALD>()) {
      mscale = 1;
      dscale = 0.5f;
      pscale = 0.5f;
      uscale = 0.5f;
   } else {
      dscale *= 0.5f;
      pscale *= 0.5f;
      uscale *= 0.5f;
   }


   // C-C
   {
      real coef1 = bn[0];
      real coef3 = bn[1] * r;
      // phi_c c
      phi1[0] += coef1 * ck;
      phi2[0] += coef1 * ci;
      phi1z[0] += coef3 * ck;
   }


   // D-C and C-D
   {
      real coef3 = bn[1] * r;
      real coef5 = (bn[1] - bn[2] * r2);
      // phi_d c
      phi1[0] += -coef3 * dk.z;
      phi2[0] += coef3 * di.z;
      phi1z[0] += coef5 * dk.z;
      // dphi_c d
      // phi1[1]; phi1[2];
      phi1[3] += coef3 * ck;
      // phi2[1]; phi2[2];
      phi2[3] += -coef3 * ci;
      // phi1z[1]; phi1z[2];
      phi1z[3] += -coef5 * ck;
   }


   // D-D
   {
      real coef3 = bn[1];
      real coef5 = (bn[1] - bn[2] * r2);
      real coez5 = bn[2] * r;
      real coez7 = (3 * bn[2] - bn[3] * r2) * r;
      // dphi_d d
      phi1[1] += coef3 * dk.x;
      phi1[2] += coef3 * dk.y;
      phi1[3] += coef5 * dk.z;
      phi2[1] += coef3 * di.x;
      phi2[2] += coef3 * di.y;
      phi2[3] += coef5 * di.z;
      phi1z[1] += coez5 * dk.x;
      phi1z[2] += coez5 * dk.y;
      phi1z[3] += coez7 * dk.z;
   }


   // Q-C and C-Q
   {
      real coef3 = bn[1];
      real coef5 = bn[2] * r2;
      real coez5 = bn[2] * r;
      real coez7 = bn[3] * r2 * r;
      // phi_q c
      phi1[0] += coef5 * qkzz;
      phi2[0] += coef5 * qizz;
      phi1z[0] += -(2 * coez5 - coez7) * qkzz;
      // d2phi_c q
      phi1[4] += -coef3 * ck;
      phi1[5] += -coef3 * ck;
      phi1[6] += -(coef3 - coef5) * ck;
      // phi1[7]; phi1[8]; phi1[9];
      phi2[4] += -coef3 * ci;
      phi2[5] += -coef3 * ci;
      phi2[6] += -(coef3 - coef5) * ci;
      // phi2[7]; phi2[8]; phi2[9];
      phi1z[4] += -coez5 * ck;
      phi1z[5] += -coez5 * ck;
      phi1z[6] += -(3 * coez5 - coez7) * ck;
      // phi1z[7]; phi1z[8]; phi1z[9];
   }


   // Q-D and D-Q
   {
      real coef5 = bn[2] * r;
      real coef7 = bn[3] * r2 * r;
      real coez7 = (bn[2] - bn[3] * r2);
      real coez9 = (3 * bn[3] - bn[4] * r2) * r2;
      // dphi_q d
      phi1[1] += -2 * coef5 * qkxz;
      phi1[2] += -2 * coef5 * qkyz;
      phi1[3] += -(2 * coef5 - coef7) * qkzz;
      phi2[1] += 2 * coef5 * qixz;
      phi2[2] += 2 * coef5 * qiyz;
      phi2[3] += (2 * coef5 - coef7) * qizz;
      phi1z[1] += 2 * coez7 * qkxz;
      phi1z[2] += 2 * coez7 * qkyz;
      phi1z[3] += (2 * coez7 - coez9) * qkzz;
      // d2phi_d q
      phi1[4] += coef5 * dk.z;
      phi1[5] += coef5 * dk.z;
      phi1[6] += (3 * coef5 - coef7) * dk.z;
      // phi1[7];
      phi1[8] += 2 * coef5 * dk.x;
      phi1[9] += 2 * coef5 * dk.y;
      //
      phi2[4] += -coef5 * di.z;
      phi2[5] += -coef5 * di.z;
      phi2[6] += -(3 * coef5 - coef7) * di.z;
      // phi2[7];
      phi2[8] += -2 * coef5 * di.x;
      phi2[9] += -2 * coef5 * di.y;
      //
      phi1z[4] += -coez7 * dk.z;
      phi1z[5] += -coez7 * dk.z;
      phi1z[6] += -(3 * coez7 - coez9) * dk.z;
      // phi1z[7];
      phi1z[8] += -2 * coez7 * dk.x;
      phi1z[9] += -2 * coez7 * dk.y;
   }


   // Q-Q
   {
      // d2phi_q q
      real coef5 = bn[2];
      real coef7 = bn[3] * r2;
      real coef9 = bn[4] * r2 * r2;
      real coez7 = bn[3] * r;
      real coez9 = bn[4] * r2 * r;
      real coez11 = bn[5] * r2 * r2 * r;
      //
      phi1[4] += 2 * coef5 * qkxx - coef7 * qkzz;
      phi1[5] += 2 * coef5 * qkyy - coef7 * qkzz;
      phi1[6] += (2 * coef5 - 5 * coef7 + coef9) * qkzz;
      phi1[7] += 4 * coef5 * qkxy;
      phi1[8] += 4 * (coef5 - coef7) * qkxz;
      phi1[9] += 4 * (coef5 - coef7) * qkyz;
      //
      phi2[4] += 2 * coef5 * qixx - coef7 * qizz;
      phi2[5] += 2 * coef5 * qiyy - coef7 * qizz;
      phi2[6] += (2 * coef5 - 5 * coef7 + coef9) * qizz;
      phi2[7] += 4 * coef5 * qixy;
      phi2[8] += 4 * (coef5 - coef7) * qixz;
      phi2[9] += 4 * (coef5 - coef7) * qiyz;
      //
      phi1z[4] += 2 * coez7 * qkxx + (2 * coez7 - coez9) * qkzz;
      phi1z[5] += 2 * coez7 * qkyy + (2 * coez7 - coez9) * qkzz;
      phi1z[6] += (12 * coez7 - 9 * coez9 + coez11) * qkzz;
      phi1z[7] += 4 * coez7 * qkxy;
      phi1z[8] += 4 * (3 * coez7 - coez9) * qkxz;
      phi1z[9] += 4 * (3 * coez7 - coez9) * qkyz;
   }


   #pragma unroll
   for (int i = 0; i < 10; ++i) {
      phi1[i] *= mscale;
      phi2[i] *= mscale;
      phi1z[i] *= mscale;
   }


   if CONSTEXPR (do_e) {
      real e = phi1[0] * ci + phi1[1] * di.x + phi1[2] * di.y + phi1[3] * di.z +
         phi1[4] * qixx + phi1[5] * qiyy + phi1[6] * qizz + phi1[7] * qixy +
         phi1[8] * qixz + phi1[9] * qiyz;
      eo = f * e;
   }


   real phi1d[3] = {0};
   real phi2d[3] = {0};
   real phi1dz[3] = {0};


   // U-C and C-U
   {
      real coe3 = sr3 * r;
      real coe5 = sr3 - sr5 * r2;
      real coed3 = dscale * coe3;
      real coed5 = dscale * coe5;
      real coep3 = pscale * coe3;
      real coep5 = pscale * coe5;
      // phi_u c
      phi1[0] += -(coed3 * ukp.z + coep3 * ukd.z);
      phi2[0] += coed3 * uip.z + coep3 * uid.z;
      phi1z[0] += coed5 * ukp.z + coep5 * ukd.z;
      // dphi_c u
      phi1d[2] += coe3 * ck;
      phi2d[2] += -coe3 * ci;
      phi1dz[2] += -coe5 * ck;
   }


   // U-D and D-U
   {
      real coe3 = sr3;
      real coe5 = sr5 * r2;
      real coez5 = sr5 * r;
      real coez7 = sr7 * r2 * r;
      real coed3 = dscale * coe3;
      real coed5 = dscale * coe5;
      real coedz5 = dscale * coez5;
      real coedz7 = dscale * coez7;
      real coep3 = pscale * coe3;
      real coep5 = pscale * coe5;
      real coepz5 = pscale * coez5;
      real coepz7 = pscale * coez7;
      // dphi_u d
      phi1[1] += coed3 * ukp.x + coep3 * ukd.x;
      phi1[2] += coed3 * ukp.y + coep3 * ukd.y;
      phi1[3] += (coed3 - coed5) * ukp.z + (coep3 - coep5) * ukd.z;
      phi2[1] += coed3 * uip.x + coep3 * uid.x;
      phi2[2] += coed3 * uip.y + coep3 * uid.y;
      phi2[3] += (coed3 - coed5) * uip.z + (coep3 - coep5) * uid.z;
      phi1z[1] += coedz5 * ukp.x + coepz5 * ukd.x;
      phi1z[2] += coedz5 * ukp.y + coepz5 * ukd.y;
      phi1z[3] += (3 * coedz5 - coedz7) * ukp.z + (3 * coepz5 - coepz7) * ukd.z;
      // dphi_d u
      phi1d[0] += coe3 * dk.x;
      phi1d[1] += coe3 * dk.y;
      phi1d[2] += (coe3 - coe5) * dk.z;
      phi2d[0] += coe3 * di.x;
      phi2d[1] += coe3 * di.y;
      phi2d[2] += (coe3 - coe5) * di.z;
      phi1dz[0] += coez5 * dk.x;
      phi1dz[1] += coez5 * dk.y;
      phi1dz[2] += (3 * coez5 - coez7) * dk.z;
   }


   // U-Q and Q-U
   {
      real coe5 = sr5 * r;
      real coe7 = sr7 * r2 * r;
      real coez7 = sr5 - sr7 * r2;
      real coez9 = (3 * sr7 - sr9 * r2) * r2;
      real coed5 = dscale * coe5;
      real coed7 = dscale * coe7;
      real coedz7 = dscale * coez7;
      real coedz9 = dscale * coez9;
      real coep5 = pscale * coe5;
      real coep7 = pscale * coe7;
      real coepz7 = pscale * coez7;
      real coepz9 = pscale * coez9;
      // d2phi_u q
      phi1[4] += coed5 * ukp.z + coep5 * ukd.z;
      phi1[5] += coed5 * ukp.z + coep5 * ukd.z;
      phi1[6] += (3 * coed5 - coed7) * ukp.z + (3 * coep5 - coep7) * ukd.z;
      // phi1[7];
      phi1[8] += 2 * (coed5 * ukp.x + coep5 * ukd.x);
      phi1[9] += 2 * (coed5 * ukp.y + coep5 * ukd.y);
      //
      phi2[4] += -(coed5 * uip.z + coep5 * uid.z);
      phi2[5] += -(coed5 * uip.z + coep5 * uid.z);
      phi2[6] += -(3 * coed5 - coed7) * uip.z - (3 * coep5 - coep7) * uid.z;
      // phi2[7];
      phi2[8] += -2 * (coed5 * uip.x + coep5 * uid.x);
      phi2[9] += -2 * (coed5 * uip.y + coep5 * uid.y);
      //
      phi1z[4] += -(coedz7 * ukp.z + coepz7 * ukd.z);
      phi1z[5] += -(coedz7 * ukp.z + coepz7 * ukd.z);
      phi1z[6] +=
         -(3 * coedz7 - coedz9) * ukp.z - (3 * coepz7 - coepz9) * ukd.z;
      // phi1z[7];
      phi1z[8] += -2 * (coedz7 * ukp.x + coepz7 * ukd.x);
      phi1z[9] += -2 * (coedz7 * ukp.y + coepz7 * ukd.y);
      // dphi_q u
      phi1d[0] += -2 * coe5 * qkxz;
      phi1d[1] += -2 * coe5 * qkyz;
      phi1d[2] += -(2 * coe5 - coe7) * qkzz;
      phi2d[0] += 2 * coe5 * qixz;
      phi2d[1] += 2 * coe5 * qiyz;
      phi2d[2] += (2 * coe5 - coe7) * qizz;
      phi1dz[0] += 2 * coez7 * qkxz;
      phi1dz[1] += 2 * coez7 * qkyz;
      phi1dz[2] += (2 * coez7 - coez9) * qkzz;
   }


   real3 frc, trq1, trq2;
   if CONSTEXPR (do_g) {
      // torque
      real3 trqa = cross(phi1[1], phi1[2], phi1[3], di);
      trqa.x += phi1[9] * (qizz - qiyy) + 2 * (phi1[5] - phi1[6]) * qiyz +
         phi1[7] * qixz - phi1[8] * qixy;
      trqa.y += phi1[8] * (qixx - qizz) + 2 * (phi1[6] - phi1[4]) * qixz +
         phi1[9] * qixy - phi1[7] * qiyz;
      trqa.z += phi1[7] * (qiyy - qixx) + 2 * (phi1[4] - phi1[5]) * qixy +
         phi1[8] * qiyz - phi1[9] * qixz;
      real3 trqb = cross(phi2[1], phi2[2], phi2[3], dk);
      trqb.x += phi2[9] * (qkzz - qkyy) + 2 * (phi2[5] - phi2[6]) * qkyz +
         phi2[7] * qkxz - phi2[8] * qkxy;
      trqb.y += phi2[8] * (qkxx - qkzz) + 2 * (phi2[6] - phi2[4]) * qkxz +
         phi2[9] * qkxy - phi2[7] * qkyz;
      trqb.z += phi2[7] * (qkyy - qkxx) + 2 * (phi2[4] - phi2[5]) * qkxy +
         phi2[8] * qkyz - phi2[9] * qkxz;
      trq1 = trqa;
      trq2 = trqb;


      real3 trqau =
         cross(phi1d[0], phi1d[1], phi1d[2], (dscale * uip + pscale * uid));
      real3 trqbu =
         cross(phi2d[0], phi2d[1], phi2d[2], (dscale * ukp + pscale * ukd));


      // gradient
      real frc1z = phi1z[0] * ci + phi1z[1] * di.x + phi1z[2] * di.y +
         phi1z[3] * di.z + phi1z[4] * qixx + phi1z[5] * qiyy + phi1z[6] * qizz +
         phi1z[7] * qixy + phi1z[8] * qixz + phi1z[9] * qiyz;
      frc1z +=
         dot3(phi1dz[0], phi1dz[1], phi1dz[2], (dscale * uip + pscale * uid));
      frc.x = -invr1 * (trqa.y + trqb.y + trqau.y + trqbu.y);
      frc.y = invr1 * (trqa.x + trqb.x + trqau.x + trqbu.x);
      frc.z = frc1z;
   }


   // U-U
   {
      real coeu5 = uscale * sr5 * r;
      real coeu7 = uscale * sr7 * r2 * r;
      frc.x += coeu5 *
         (uid.x * ukp.z + uid.z * ukp.x + uip.x * ukd.z + uip.z * ukd.x);
      frc.y += coeu5 *
         (uid.y * ukp.z + uid.z * ukp.y + uip.y * ukd.z + uip.z * ukd.y);
      frc.z += coeu5 *
            (uid.x * ukp.x + uid.y * ukp.y + uip.x * ukd.x + uip.y * ukd.y) +
         (3 * coeu5 - coeu7) * (uid.z * ukp.z + uip.z * ukd.z);
   }


   if CONSTEXPR (do_g) {
      real3 glfrc;
      rotQI2GVector(rot, frc, glfrc);
      frc = f * glfrc;
      frcxi += frc.x;
      frcyi += frc.y;
      frczi += frc.z;
      frcxk -= frc.x;
      frcyk -= frc.y;
      frczk -= frc.z;
      real3 gltrq1;
      rotQI2GVector(rot, trq1, gltrq1);
      trqxi += f * gltrq1.x;
      trqyi += f * gltrq1.y;
      trqzi += f * gltrq1.z;
      real3 gltrq2;
      rotQI2GVector(rot, trq2, gltrq2);
      trqxk += f * gltrq2.x;
      trqyk += f * gltrq2.y;
      trqzk += f * gltrq2.z;
   }
   if CONSTEXPR (do_v) {
      voxx = -dR.x * frc.x;
      voxy = -0.5f * (dR.y * frc.x + dR.x * frc.y);
      voxz = -0.5f * (dR.z * frc.x + dR.x * frc.z);
      voyy = -dR.y * frc.y;
      voyz = -0.5f * (dR.z * frc.y + dR.y * frc.z);
      vozz = -dR.z * frc.z;
   }
}


// ck.py Version 2.0.1


template <class Ver, class ETYP>
__global__
void emplar_cu1c(TINKER_IMAGE_PARAMS, energy_buffer restrict ebuf,
                 virial_buffer restrict vbuf, grad_prec* restrict gx,
                 grad_prec* restrict gy, grad_prec* restrict gz, real off,
                 real* restrict trqx, real* restrict trqy, real* restrict trqz,
                 const real (*restrict rpole)[10],
                 const real (*restrict uind)[3], const real (*restrict uinp)[3],
                 const real* restrict thole, const real* restrict pdamp, real f,
                 real aewald, int nexclude, const int (*restrict exclude)[2],
                 const real (*restrict exclude_scale)[4],
                 const real* restrict x, const real* restrict y,
                 const real* restrict z)
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_g = Ver::g;
   constexpr bool do_v = Ver::v;
   static_assert(!Ver::a, "");
   const int ithread = threadIdx.x + blockIdx.x * blockDim.x;


   using ebuf_prec = energy_buffer_traits::type;
   ebuf_prec ebuftl;
   if CONSTEXPR (do_e) {
      ebuftl = 0;
   }
   using vbuf_prec = virial_buffer_traits::type;
   vbuf_prec vbuftlxx, vbuftlyx, vbuftlzx, vbuftlyy, vbuftlzy, vbuftlzz;
   if CONSTEXPR (do_v) {
      vbuftlxx = 0;
      vbuftlyx = 0;
      vbuftlzx = 0;
      vbuftlyy = 0;
      vbuftlzy = 0;
      vbuftlzz = 0;
   }
   __shared__ real xi[BLOCK_DIM];
   __shared__ real yi[BLOCK_DIM];
   __shared__ real zi[BLOCK_DIM];
   real xk;
   real yk;
   real zk;
   __shared__ real frcxi[BLOCK_DIM];
   __shared__ real frcyi[BLOCK_DIM];
   __shared__ real frczi[BLOCK_DIM];
   __shared__ real trqxi[BLOCK_DIM];
   __shared__ real trqyi[BLOCK_DIM];
   __shared__ real trqzi[BLOCK_DIM];
   real frcxk;
   real frcyk;
   real frczk;
   real trqxk;
   real trqyk;
   real trqzk;
   __shared__ real ci[BLOCK_DIM];
   __shared__ real dix[BLOCK_DIM];
   __shared__ real diy[BLOCK_DIM];
   __shared__ real diz[BLOCK_DIM];
   __shared__ real qixx[BLOCK_DIM];
   __shared__ real qixy[BLOCK_DIM];
   __shared__ real qixz[BLOCK_DIM];
   __shared__ real qiyy[BLOCK_DIM];
   __shared__ real qiyz[BLOCK_DIM];
   __shared__ real qizz[BLOCK_DIM];
   __shared__ real uidx[BLOCK_DIM];
   __shared__ real uidy[BLOCK_DIM];
   __shared__ real uidz[BLOCK_DIM];
   __shared__ real uipx[BLOCK_DIM];
   __shared__ real uipy[BLOCK_DIM];
   __shared__ real uipz[BLOCK_DIM];
   __shared__ real pdi[BLOCK_DIM];
   __shared__ real pti[BLOCK_DIM];
   real ck;
   real dkx;
   real dky;
   real dkz;
   real qkxx;
   real qkxy;
   real qkxz;
   real qkyy;
   real qkyz;
   real qkzz;
   real ukdx;
   real ukdy;
   real ukdz;
   real ukpx;
   real ukpy;
   real ukpz;
   real pdk;
   real ptk;


   for (int ii = ithread; ii < nexclude; ii += blockDim.x * gridDim.x) {
      const int klane = threadIdx.x;
      if CONSTEXPR (do_g) {
         frcxi[threadIdx.x] = 0;
         frcyi[threadIdx.x] = 0;
         frczi[threadIdx.x] = 0;
         trqxi[threadIdx.x] = 0;
         trqyi[threadIdx.x] = 0;
         trqzi[threadIdx.x] = 0;
         frcxk = 0;
         frcyk = 0;
         frczk = 0;
         trqxk = 0;
         trqyk = 0;
         trqzk = 0;
      }


      int i = exclude[ii][0];
      int k = exclude[ii][1];
      real scalea = exclude_scale[ii][0];
      real scaleb = exclude_scale[ii][1];
      real scalec = exclude_scale[ii][2];
      real scaled = exclude_scale[ii][3];


      xi[klane] = x[i];
      yi[klane] = y[i];
      zi[klane] = z[i];
      xk = x[k];
      yk = y[k];
      zk = z[k];
      ci[klane] = rpole[i][mpl_pme_0];
      dix[klane] = rpole[i][mpl_pme_x];
      diy[klane] = rpole[i][mpl_pme_y];
      diz[klane] = rpole[i][mpl_pme_z];
      qixx[klane] = rpole[i][mpl_pme_xx];
      qixy[klane] = rpole[i][mpl_pme_xy];
      qixz[klane] = rpole[i][mpl_pme_xz];
      qiyy[klane] = rpole[i][mpl_pme_yy];
      qiyz[klane] = rpole[i][mpl_pme_yz];
      qizz[klane] = rpole[i][mpl_pme_zz];
      uidx[klane] = uind[i][0];
      uidy[klane] = uind[i][1];
      uidz[klane] = uind[i][2];
      uipx[klane] = uinp[i][0];
      uipy[klane] = uinp[i][1];
      uipz[klane] = uinp[i][2];
      pdi[klane] = pdamp[i];
      pti[klane] = thole[i];
      ck = rpole[k][mpl_pme_0];
      dkx = rpole[k][mpl_pme_x];
      dky = rpole[k][mpl_pme_y];
      dkz = rpole[k][mpl_pme_z];
      qkxx = rpole[k][mpl_pme_xx];
      qkxy = rpole[k][mpl_pme_xy];
      qkxz = rpole[k][mpl_pme_xz];
      qkyy = rpole[k][mpl_pme_yy];
      qkyz = rpole[k][mpl_pme_yz];
      qkzz = rpole[k][mpl_pme_zz];
      ukdx = uind[k][0];
      ukdy = uind[k][1];
      ukdz = uind[k][2];
      ukpx = uinp[k][0];
      ukpy = uinp[k][1];
      ukpz = uinp[k][2];
      pdk = pdamp[k];
      ptk = thole[k];


      constexpr bool incl = true;
      real xr = xk - xi[klane];
      real yr = yk - yi[klane];
      real zr = zk - zi[klane];
      real r2 = image2(xr, yr, zr);
      if (r2 <= off * off and incl) {
         real e1, vxx1, vyx1, vzx1, vyy1, vzy1, vzz1;
         pair_mplar<Ver, NON_EWALD>(
            r2, make_real3(xr, yr, zr), scalea - 1, scaleb - 1, scalec - 1,
            scaled - 1, ci[klane],
            make_real3(dix[klane], diy[klane], diz[klane]), qixx[klane],
            qixy[klane], qixz[klane], qiyy[klane], qiyz[klane], qizz[klane],
            make_real3(uidx[klane], uidy[klane], uidz[klane]),
            make_real3(uipx[klane], uipy[klane], uipz[klane]), pdi[klane],
            pti[klane], ck, make_real3(dkx, dky, dkz), qkxx, qkxy, qkxz, qkyy,
            qkyz, qkzz, make_real3(ukdx, ukdy, ukdz),
            make_real3(ukpx, ukpy, ukpz), pdk, ptk, f, aewald, frcxi[klane],
            frcyi[klane], frczi[klane], frcxk, frcyk, frczk, trqxi[klane],
            trqyi[klane], trqzi[klane], trqxk, trqyk, trqzk, e1, vxx1, vyx1,
            vzx1, vyy1, vzy1, vzz1);
         if CONSTEXPR (do_e) {
            ebuftl += cvt_to<ebuf_prec>(e1);
         }
         if CONSTEXPR (do_v) {
            vbuftlxx += cvt_to<vbuf_prec>(vxx1);
            vbuftlyx += cvt_to<vbuf_prec>(vyx1);
            vbuftlzx += cvt_to<vbuf_prec>(vzx1);
            vbuftlyy += cvt_to<vbuf_prec>(vyy1);
            vbuftlzy += cvt_to<vbuf_prec>(vzy1);
            vbuftlzz += cvt_to<vbuf_prec>(vzz1);
         }
      } // end if (include)


      if CONSTEXPR (do_g) {
         atomic_add(frcxi[threadIdx.x], gx, i);
         atomic_add(frcyi[threadIdx.x], gy, i);
         atomic_add(frczi[threadIdx.x], gz, i);
         atomic_add(trqxi[threadIdx.x], trqx, i);
         atomic_add(trqyi[threadIdx.x], trqy, i);
         atomic_add(trqzi[threadIdx.x], trqz, i);
         atomic_add(frcxk, gx, k);
         atomic_add(frcyk, gy, k);
         atomic_add(frczk, gz, k);
         atomic_add(trqxk, trqx, k);
         atomic_add(trqyk, trqy, k);
         atomic_add(trqzk, trqz, k);
      }
   }


   if CONSTEXPR (do_e) {
      atomic_add(ebuftl, ebuf, ithread);
   }
   if CONSTEXPR (do_v) {
      atomic_add(vbuftlxx, vbuftlyx, vbuftlzx, vbuftlyy, vbuftlzy, vbuftlzz,
                 vbuf, ithread);
   }
}


template <class Ver, class ETYP>
__global__
void emplar_cu1b(TINKER_IMAGE_PARAMS, energy_buffer restrict ebuf,
                 virial_buffer restrict vbuf, grad_prec* restrict gx,
                 grad_prec* restrict gy, grad_prec* restrict gz, real off,
                 real* restrict trqx, real* restrict trqy, real* restrict trqz,
                 const real (*restrict rpole)[10],
                 const real (*restrict uind)[3], const real (*restrict uinp)[3],
                 const real* restrict thole, const real* restrict pdamp, real f,
                 real aewald, const Spatial::SortedAtom* restrict sorted, int n,
                 int nakpl, const int* restrict iakpl)
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_g = Ver::g;
   constexpr bool do_v = Ver::v;
   static_assert(!Ver::a, "");
   const int ithread = threadIdx.x + blockIdx.x * blockDim.x;
   const int iwarp = ithread / WARP_SIZE;
   const int nwarp = blockDim.x * gridDim.x / WARP_SIZE;
   const int ilane = threadIdx.x & (WARP_SIZE - 1);


   using ebuf_prec = energy_buffer_traits::type;
   ebuf_prec ebuftl;
   if CONSTEXPR (do_e) {
      ebuftl = 0;
   }
   using vbuf_prec = virial_buffer_traits::type;
   vbuf_prec vbuftlxx, vbuftlyx, vbuftlzx, vbuftlyy, vbuftlzy, vbuftlzz;
   if CONSTEXPR (do_v) {
      vbuftlxx = 0;
      vbuftlyx = 0;
      vbuftlzx = 0;
      vbuftlyy = 0;
      vbuftlzy = 0;
      vbuftlzz = 0;
   }
   __shared__ real xi[BLOCK_DIM];
   __shared__ real yi[BLOCK_DIM];
   __shared__ real zi[BLOCK_DIM];
   real xk;
   real yk;
   real zk;
   __shared__ real frcxi[BLOCK_DIM];
   __shared__ real frcyi[BLOCK_DIM];
   __shared__ real frczi[BLOCK_DIM];
   __shared__ real trqxi[BLOCK_DIM];
   __shared__ real trqyi[BLOCK_DIM];
   __shared__ real trqzi[BLOCK_DIM];
   real frcxk;
   real frcyk;
   real frczk;
   real trqxk;
   real trqyk;
   real trqzk;
   __shared__ real ci[BLOCK_DIM];
   __shared__ real dix[BLOCK_DIM];
   __shared__ real diy[BLOCK_DIM];
   __shared__ real diz[BLOCK_DIM];
   __shared__ real qixx[BLOCK_DIM];
   __shared__ real qixy[BLOCK_DIM];
   __shared__ real qixz[BLOCK_DIM];
   __shared__ real qiyy[BLOCK_DIM];
   __shared__ real qiyz[BLOCK_DIM];
   __shared__ real qizz[BLOCK_DIM];
   __shared__ real uidx[BLOCK_DIM];
   __shared__ real uidy[BLOCK_DIM];
   __shared__ real uidz[BLOCK_DIM];
   __shared__ real uipx[BLOCK_DIM];
   __shared__ real uipy[BLOCK_DIM];
   __shared__ real uipz[BLOCK_DIM];
   __shared__ real pdi[BLOCK_DIM];
   __shared__ real pti[BLOCK_DIM];
   real ck;
   real dkx;
   real dky;
   real dkz;
   real qkxx;
   real qkxy;
   real qkxz;
   real qkyy;
   real qkyz;
   real qkzz;
   real ukdx;
   real ukdy;
   real ukdz;
   real ukpx;
   real ukpy;
   real ukpz;
   real pdk;
   real ptk;


   for (int iw = iwarp; iw < nakpl; iw += nwarp) {
      if CONSTEXPR (do_g) {
         frcxi[threadIdx.x] = 0;
         frcyi[threadIdx.x] = 0;
         frczi[threadIdx.x] = 0;
         trqxi[threadIdx.x] = 0;
         trqyi[threadIdx.x] = 0;
         trqzi[threadIdx.x] = 0;
         frcxk = 0;
         frcyk = 0;
         frczk = 0;
         trqxk = 0;
         trqyk = 0;
         trqzk = 0;
      }


      int tri, tx, ty;
      tri = iakpl[iw];
      tri_to_xy(tri, tx, ty);


      int iid = ty * WARP_SIZE + ilane;
      int atomi = min(iid, n - 1);
      int i = sorted[atomi].unsorted;
      int kid = tx * WARP_SIZE + ilane;
      int atomk = min(kid, n - 1);
      int k = sorted[atomk].unsorted;
      xi[threadIdx.x] = sorted[atomi].x;
      yi[threadIdx.x] = sorted[atomi].y;
      zi[threadIdx.x] = sorted[atomi].z;
      xk = sorted[atomk].x;
      yk = sorted[atomk].y;
      zk = sorted[atomk].z;


      ci[threadIdx.x] = rpole[i][mpl_pme_0];
      dix[threadIdx.x] = rpole[i][mpl_pme_x];
      diy[threadIdx.x] = rpole[i][mpl_pme_y];
      diz[threadIdx.x] = rpole[i][mpl_pme_z];
      qixx[threadIdx.x] = rpole[i][mpl_pme_xx];
      qixy[threadIdx.x] = rpole[i][mpl_pme_xy];
      qixz[threadIdx.x] = rpole[i][mpl_pme_xz];
      qiyy[threadIdx.x] = rpole[i][mpl_pme_yy];
      qiyz[threadIdx.x] = rpole[i][mpl_pme_yz];
      qizz[threadIdx.x] = rpole[i][mpl_pme_zz];
      uidx[threadIdx.x] = uind[i][0];
      uidy[threadIdx.x] = uind[i][1];
      uidz[threadIdx.x] = uind[i][2];
      uipx[threadIdx.x] = uinp[i][0];
      uipy[threadIdx.x] = uinp[i][1];
      uipz[threadIdx.x] = uinp[i][2];
      pdi[threadIdx.x] = pdamp[i];
      pti[threadIdx.x] = thole[i];
      ck = rpole[k][mpl_pme_0];
      dkx = rpole[k][mpl_pme_x];
      dky = rpole[k][mpl_pme_y];
      dkz = rpole[k][mpl_pme_z];
      qkxx = rpole[k][mpl_pme_xx];
      qkxy = rpole[k][mpl_pme_xy];
      qkxz = rpole[k][mpl_pme_xz];
      qkyy = rpole[k][mpl_pme_yy];
      qkyz = rpole[k][mpl_pme_yz];
      qkzz = rpole[k][mpl_pme_zz];
      ukdx = uind[k][0];
      ukdy = uind[k][1];
      ukdz = uind[k][2];
      ukpx = uinp[k][0];
      ukpy = uinp[k][1];
      ukpz = uinp[k][2];
      pdk = pdamp[k];
      ptk = thole[k];


      for (int j = 0; j < WARP_SIZE; ++j) {
         int srclane = (ilane + j) & (WARP_SIZE - 1);
         int klane = srclane + threadIdx.x - ilane;
         bool incl = iid < kid and kid < n;
         real xr = xk - xi[klane];
         real yr = yk - yi[klane];
         real zr = zk - zi[klane];
         real r2 = image2(xr, yr, zr);
         if (r2 <= off * off and incl) {
            real e, vxx, vyx, vzx, vyy, vzy, vzz;
            pair_mplar<Ver, ETYP>(
               r2, make_real3(xr, yr, zr), 1, 1, 1, 1, ci[klane],
               make_real3(dix[klane], diy[klane], diz[klane]), qixx[klane],
               qixy[klane], qixz[klane], qiyy[klane], qiyz[klane], qizz[klane],
               make_real3(uidx[klane], uidy[klane], uidz[klane]),
               make_real3(uipx[klane], uipy[klane], uipz[klane]), pdi[klane],
               pti[klane], ck, make_real3(dkx, dky, dkz), qkxx, qkxy, qkxz,
               qkyy, qkyz, qkzz, make_real3(ukdx, ukdy, ukdz),
               make_real3(ukpx, ukpy, ukpz), pdk, ptk, f, aewald, frcxi[klane],
               frcyi[klane], frczi[klane], frcxk, frcyk, frczk, trqxi[klane],
               trqyi[klane], trqzi[klane], trqxk, trqyk, trqzk, e, vxx, vyx,
               vzx, vyy, vzy, vzz);
            if CONSTEXPR (do_e) {
               ebuftl += cvt_to<ebuf_prec>(e);
            }
            if CONSTEXPR (do_v) {
               vbuftlxx += cvt_to<vbuf_prec>(vxx);
               vbuftlyx += cvt_to<vbuf_prec>(vyx);
               vbuftlzx += cvt_to<vbuf_prec>(vzx);
               vbuftlyy += cvt_to<vbuf_prec>(vyy);
               vbuftlzy += cvt_to<vbuf_prec>(vzy);
               vbuftlzz += cvt_to<vbuf_prec>(vzz);
            }
         } // end if (include)


         iid = __shfl_sync(ALL_LANES, iid, ilane + 1);
      }


      if CONSTEXPR (do_g) {
         atomic_add(frcxi[threadIdx.x], gx, i);
         atomic_add(frcyi[threadIdx.x], gy, i);
         atomic_add(frczi[threadIdx.x], gz, i);
         atomic_add(trqxi[threadIdx.x], trqx, i);
         atomic_add(trqyi[threadIdx.x], trqy, i);
         atomic_add(trqzi[threadIdx.x], trqz, i);
         atomic_add(frcxk, gx, k);
         atomic_add(frcyk, gy, k);
         atomic_add(frczk, gz, k);
         atomic_add(trqxk, trqx, k);
         atomic_add(trqyk, trqy, k);
         atomic_add(trqzk, trqz, k);
      }
   }


   if CONSTEXPR (do_e) {
      atomic_add(ebuftl, ebuf, ithread);
   }
   if CONSTEXPR (do_v) {
      atomic_add(vbuftlxx, vbuftlyx, vbuftlzx, vbuftlyy, vbuftlzy, vbuftlzz,
                 vbuf, ithread);
   }
}


template <class Ver, class ETYP>
__global__
void emplar_cu1a(TINKER_IMAGE_PARAMS, energy_buffer restrict ebuf,
                 virial_buffer restrict vbuf, grad_prec* restrict gx,
                 grad_prec* restrict gy, grad_prec* restrict gz, real off,
                 real* restrict trqx, real* restrict trqy, real* restrict trqz,
                 const real (*restrict rpole)[10],
                 const real (*restrict uind)[3], const real (*restrict uinp)[3],
                 const real* restrict thole, const real* restrict pdamp, real f,
                 real aewald, const Spatial::SortedAtom* restrict sorted,
                 int niak, const int* restrict iak, const int* restrict lst)
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_g = Ver::g;
   constexpr bool do_v = Ver::v;
   static_assert(!Ver::a, "");
   const int ithread = threadIdx.x + blockIdx.x * blockDim.x;
   const int iwarp = ithread / WARP_SIZE;
   const int nwarp = blockDim.x * gridDim.x / WARP_SIZE;
   const int ilane = threadIdx.x & (WARP_SIZE - 1);


   using ebuf_prec = energy_buffer_traits::type;
   ebuf_prec ebuftl;
   if CONSTEXPR (do_e) {
      ebuftl = 0;
   }
   using vbuf_prec = virial_buffer_traits::type;
   vbuf_prec vbuftlxx, vbuftlyx, vbuftlzx, vbuftlyy, vbuftlzy, vbuftlzz;
   if CONSTEXPR (do_v) {
      vbuftlxx = 0;
      vbuftlyx = 0;
      vbuftlzx = 0;
      vbuftlyy = 0;
      vbuftlzy = 0;
      vbuftlzz = 0;
   }
   __shared__ real xi[BLOCK_DIM];
   __shared__ real yi[BLOCK_DIM];
   __shared__ real zi[BLOCK_DIM];
   real xk;
   real yk;
   real zk;
   __shared__ real frcxi[BLOCK_DIM];
   __shared__ real frcyi[BLOCK_DIM];
   __shared__ real frczi[BLOCK_DIM];
   __shared__ real trqxi[BLOCK_DIM];
   __shared__ real trqyi[BLOCK_DIM];
   __shared__ real trqzi[BLOCK_DIM];
   real frcxk;
   real frcyk;
   real frczk;
   real trqxk;
   real trqyk;
   real trqzk;
   __shared__ real ci[BLOCK_DIM];
   __shared__ real dix[BLOCK_DIM];
   __shared__ real diy[BLOCK_DIM];
   __shared__ real diz[BLOCK_DIM];
   __shared__ real qixx[BLOCK_DIM];
   __shared__ real qixy[BLOCK_DIM];
   __shared__ real qixz[BLOCK_DIM];
   __shared__ real qiyy[BLOCK_DIM];
   __shared__ real qiyz[BLOCK_DIM];
   __shared__ real qizz[BLOCK_DIM];
   __shared__ real uidx[BLOCK_DIM];
   __shared__ real uidy[BLOCK_DIM];
   __shared__ real uidz[BLOCK_DIM];
   __shared__ real uipx[BLOCK_DIM];
   __shared__ real uipy[BLOCK_DIM];
   __shared__ real uipz[BLOCK_DIM];
   __shared__ real pdi[BLOCK_DIM];
   __shared__ real pti[BLOCK_DIM];
   real ck;
   real dkx;
   real dky;
   real dkz;
   real qkxx;
   real qkxy;
   real qkxz;
   real qkyy;
   real qkyz;
   real qkzz;
   real ukdx;
   real ukdy;
   real ukdz;
   real ukpx;
   real ukpy;
   real ukpz;
   real pdk;
   real ptk;


   for (int iw = iwarp; iw < niak; iw += nwarp) {
      if CONSTEXPR (do_g) {
         frcxi[threadIdx.x] = 0;
         frcyi[threadIdx.x] = 0;
         frczi[threadIdx.x] = 0;
         trqxi[threadIdx.x] = 0;
         trqyi[threadIdx.x] = 0;
         trqzi[threadIdx.x] = 0;
         frcxk = 0;
         frcyk = 0;
         frczk = 0;
         trqxk = 0;
         trqyk = 0;
         trqzk = 0;
      }


      int ty = iak[iw];
      int atomi = ty * WARP_SIZE + ilane;
      int i = sorted[atomi].unsorted;
      int atomk = lst[iw * WARP_SIZE + ilane];
      int k = sorted[atomk].unsorted;
      xi[threadIdx.x] = sorted[atomi].x;
      yi[threadIdx.x] = sorted[atomi].y;
      zi[threadIdx.x] = sorted[atomi].z;
      xk = sorted[atomk].x;
      yk = sorted[atomk].y;
      zk = sorted[atomk].z;


      ci[threadIdx.x] = rpole[i][mpl_pme_0];
      dix[threadIdx.x] = rpole[i][mpl_pme_x];
      diy[threadIdx.x] = rpole[i][mpl_pme_y];
      diz[threadIdx.x] = rpole[i][mpl_pme_z];
      qixx[threadIdx.x] = rpole[i][mpl_pme_xx];
      qixy[threadIdx.x] = rpole[i][mpl_pme_xy];
      qixz[threadIdx.x] = rpole[i][mpl_pme_xz];
      qiyy[threadIdx.x] = rpole[i][mpl_pme_yy];
      qiyz[threadIdx.x] = rpole[i][mpl_pme_yz];
      qizz[threadIdx.x] = rpole[i][mpl_pme_zz];
      uidx[threadIdx.x] = uind[i][0];
      uidy[threadIdx.x] = uind[i][1];
      uidz[threadIdx.x] = uind[i][2];
      uipx[threadIdx.x] = uinp[i][0];
      uipy[threadIdx.x] = uinp[i][1];
      uipz[threadIdx.x] = uinp[i][2];
      pdi[threadIdx.x] = pdamp[i];
      pti[threadIdx.x] = thole[i];
      ck = rpole[k][mpl_pme_0];
      dkx = rpole[k][mpl_pme_x];
      dky = rpole[k][mpl_pme_y];
      dkz = rpole[k][mpl_pme_z];
      qkxx = rpole[k][mpl_pme_xx];
      qkxy = rpole[k][mpl_pme_xy];
      qkxz = rpole[k][mpl_pme_xz];
      qkyy = rpole[k][mpl_pme_yy];
      qkyz = rpole[k][mpl_pme_yz];
      qkzz = rpole[k][mpl_pme_zz];
      ukdx = uind[k][0];
      ukdy = uind[k][1];
      ukdz = uind[k][2];
      ukpx = uinp[k][0];
      ukpy = uinp[k][1];
      ukpz = uinp[k][2];
      pdk = pdamp[k];
      ptk = thole[k];


      for (int j = 0; j < WARP_SIZE; ++j) {
         int srclane = (ilane + j) & (WARP_SIZE - 1);
         int klane = srclane + threadIdx.x - ilane;
         bool incl = atomk > 0;
         real xr = xk - xi[klane];
         real yr = yk - yi[klane];
         real zr = zk - zi[klane];
         real r2 = image2(xr, yr, zr);
         if (r2 <= off * off and incl) {
            real e, vxx, vyx, vzx, vyy, vzy, vzz;
            pair_mplar<Ver, ETYP>(
               r2, make_real3(xr, yr, zr), 1, 1, 1, 1, ci[klane],
               make_real3(dix[klane], diy[klane], diz[klane]), qixx[klane],
               qixy[klane], qixz[klane], qiyy[klane], qiyz[klane], qizz[klane],
               make_real3(uidx[klane], uidy[klane], uidz[klane]),
               make_real3(uipx[klane], uipy[klane], uipz[klane]), pdi[klane],
               pti[klane], ck, make_real3(dkx, dky, dkz), qkxx, qkxy, qkxz,
               qkyy, qkyz, qkzz, make_real3(ukdx, ukdy, ukdz),
               make_real3(ukpx, ukpy, ukpz), pdk, ptk, f, aewald, frcxi[klane],
               frcyi[klane], frczi[klane], frcxk, frcyk, frczk, trqxi[klane],
               trqyi[klane], trqzi[klane], trqxk, trqyk, trqzk, e, vxx, vyx,
               vzx, vyy, vzy, vzz);
            if CONSTEXPR (do_e) {
               ebuftl += cvt_to<ebuf_prec>(e);
            }
            if CONSTEXPR (do_v) {
               vbuftlxx += cvt_to<vbuf_prec>(vxx);
               vbuftlyx += cvt_to<vbuf_prec>(vyx);
               vbuftlzx += cvt_to<vbuf_prec>(vzx);
               vbuftlyy += cvt_to<vbuf_prec>(vyy);
               vbuftlzy += cvt_to<vbuf_prec>(vzy);
               vbuftlzz += cvt_to<vbuf_prec>(vzz);
            }
         } // end if (include)
      }


      if CONSTEXPR (do_g) {
         atomic_add(frcxi[threadIdx.x], gx, i);
         atomic_add(frcyi[threadIdx.x], gy, i);
         atomic_add(frczi[threadIdx.x], gz, i);
         atomic_add(trqxi[threadIdx.x], trqx, i);
         atomic_add(trqyi[threadIdx.x], trqy, i);
         atomic_add(trqzi[threadIdx.x], trqz, i);
         atomic_add(frcxk, gx, k);
         atomic_add(frcyk, gy, k);
         atomic_add(frczk, gz, k);
         atomic_add(trqxk, trqx, k);
         atomic_add(trqyk, trqy, k);
         atomic_add(trqzk, trqz, k);
      }
   }


   if CONSTEXPR (do_e) {
      atomic_add(ebuftl, ebuf, ithread);
   }
   if CONSTEXPR (do_v) {
      atomic_add(vbuftlxx, vbuftlyx, vbuftlzx, vbuftlyy, vbuftlzy, vbuftlzz,
                 vbuf, ithread);
   }
}


template <class Ver, class ETYP>
void emplar_cu(const real (*uind)[3], const real (*uinp)[3])
{
   const auto& st = *mspatial_v2_unit;
   real off;
   if CONSTEXPR (eq<ETYP, EWALD>())
      off = switch_off(switch_ewald);
   else
      off = switch_off(switch_mpole);


   const real f = electric / dielec;
   real aewald = 0;
   if CONSTEXPR (eq<ETYP, EWALD>()) {
      assert(epme_unit == ppme_unit);
      PMEUnit pu = epme_unit;
      aewald = pu->aewald;


      if CONSTEXPR (Ver::e) {
         auto ker0 = empole_self_cu<Ver::a>;
         launch_k1b(g::s0, n, ker0, //
                    nullptr, em, rpole, n, f, aewald);
      }
   }
   int ngrid = get_grid_size(BLOCK_DIM);
   auto kera = emplar_cu1a<Ver, ETYP>;
   kera<<<ngrid, BLOCK_DIM, 0, g::s0>>>(
      TINKER_IMAGE_ARGS, em, vir_em, demx, demy, demz, off, trqx, trqy, trqz,
      rpole, uind, uinp, thole, pdamp, f, aewald, //
      st.sorted, st.niak, st.iak, st.lst);
   auto kerb = emplar_cu1b<Ver, ETYP>;
   kerb<<<ngrid, BLOCK_DIM, 0, g::s0>>>(
      TINKER_IMAGE_ARGS, em, vir_em, demx, demy, demz, off, trqx, trqy, trqz,
      rpole, uind, uinp, thole, pdamp, f, aewald, //
      st.sorted, st.n, st.nakpl, st.iakpl);
   auto kerc = emplar_cu1c<Ver, ETYP>;
   kerc<<<ngrid, BLOCK_DIM, 0, g::s0>>>(
      TINKER_IMAGE_ARGS, em, vir_em, demx, demy, demz, off, trqx, trqy, trqz,
      rpole, uind, uinp, thole, pdamp, f, aewald, //
      nmdpuexclude, mdpuexclude, mdpuexclude_scale, st.x, st.y, st.z);
}


template <class Ver>
void emplar_ewald_cu()
{
   // induce
   induce(uind, uinp);

   // empole real self; epolar real without epolar energy
   emplar_cu<Ver, EWALD>(uind, uinp);
   // empole recip
   empole_ewald_recip(Ver::value);
   // epolar recip self; must toggle off the calc::energy flag
   epolar_ewald_recip_self(Ver::value & ~calc::energy);

   // epolar energy
   if CONSTEXPR (Ver::e)
      epolar0_dotprod(uind, udirp);
}


template <class Ver>
void emplar_nonewald_cu()
{
   // induce
   induce(uind, uinp);

   // empole and epolar
   emplar_cu<Ver, NON_EWALD>(uind, uinp);
   if CONSTEXPR (Ver::e)
      epolar0_dotprod(uind, udirp);
}


void emplar_cu(int vers)
{
   if (use_ewald()) {
      if (vers == calc::v0)
         emplar_ewald_cu<calc::V0>();
      else if (vers == calc::v1)
         emplar_ewald_cu<calc::V1>();
      // else if (vers == calc::v3)
      //    emplar_ewald_cu<calc::V3>();
      else if (vers == calc::v4)
         emplar_ewald_cu<calc::V4>();
      else if (vers == calc::v5)
         emplar_ewald_cu<calc::V5>();
      else if (vers == calc::v6)
         emplar_ewald_cu<calc::V6>();
   } else {
      if (vers == calc::v0)
         emplar_nonewald_cu<calc::V0>();
      else if (vers == calc::v1)
         emplar_nonewald_cu<calc::V1>();
      // else if (vers == calc::v3)
      //    emplar_nonewald_cu<calc::V3>();
      else if (vers == calc::v4)
         emplar_nonewald_cu<calc::V4>();
      else if (vers == calc::v5)
         emplar_nonewald_cu<calc::V5>();
      else if (vers == calc::v6)
         emplar_nonewald_cu<calc::V6>();
   }
}
}
