#include "e_tors.h"
#include "md.h"
#include "potent.h"
#include <tinker/detail/torpot.hh>
#include <tinker/detail/tors.hh>

TINKER_NAMESPACE_BEGIN
void etors_data(rc_op op)
{
   if (!use_potent(torsion_term))
      return;

   if (op & rc_dealloc) {
      device_array::deallocate(itors, tors1, tors2, tors3, tors4, tors5, tors6);

      buffer_deallocate(et, vir_et);
   }

   if (op & rc_alloc) {
      ntors = count_bonded_term(torsion_term);
      device_array::allocate(ntors, &itors, &tors1, &tors2, &tors3, &tors4,
                             &tors5, &tors6);

      buffer_allocate(&et, &vir_et);
   }

   if (op & rc_init) {
      std::vector<int> ibuf(4 * ntors);
      for (int i = 0; i < 4 * ntors; ++i) {
         ibuf[i] = tors::itors[i] - 1;
      }
      device_array::copyin(WAIT_NEW_Q, ntors, itors, ibuf.data());
      device_array::copyin(WAIT_NEW_Q, ntors, tors1, tors::tors1);
      device_array::copyin(WAIT_NEW_Q, ntors, tors2, tors::tors2);
      device_array::copyin(WAIT_NEW_Q, ntors, tors3, tors::tors3);
      device_array::copyin(WAIT_NEW_Q, ntors, tors4, tors::tors4);
      device_array::copyin(WAIT_NEW_Q, ntors, tors5, tors::tors5);
      device_array::copyin(WAIT_NEW_Q, ntors, tors6, tors::tors6);
      torsunit = torpot::torsunit;
   }
}

void etors(int vers)
{
   extern void etors_acc(int);
   etors_acc(vers);
}
TINKER_NAMESPACE_END
