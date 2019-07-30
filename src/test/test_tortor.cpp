#include "files.h"
#include "test/ff.h"
#include "test/rt.h"
#include "test/test.h"

using namespace TINKER_NAMESPACE;
using namespace test;

static const char* tortorterm_only = R"**(
tortorterm  only
)**";

static int usage = calc::xyz | calc::vmask;

static const double ref_g_tortor_trpcage[][3] = {
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {-0.9512, 0.4126, -0.1011},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {-2.1539, -0.7023, 1.7014},
    {6.7771, 0.7133, -3.0260},  {-6.2981, -1.1089, 2.3703},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.2913, 2.8558, 0.9500},   {2.8341, -2.2584, -3.1565},
    {0.8579, -2.9278, 1.8865},  {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {-1.3573, 3.0157, -0.6247},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {-0.1406, -0.1393, 0.3421}, {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0404, 0.0228, -0.1749},
    {0.1877, 0.2741, -0.3921},  {0.1185, -0.0933, -0.0583},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {-0.2059, -0.0642, 0.2832}, {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000},
    {0.0000, 0.0000, 0.0000},   {0.0000, 0.0000, 0.0000}};
TEST_CASE("Tortor-Trpcage", "[ff][etortor][trpcage]") {
  const char* k = "test_trpcage.key";
  const char* x1 = "test_trpcage.xyz";
  const char* p = "amoebapro13.prm";

  std::string k0 = trpcage_key;
  k0 += tortorterm_only;
  file fke(k, k0);

  file fx1(x1, trpcage_xyz);
  file fpr(p, amoebapro13_prm);

  const char* argv[] = {"dummy", x1};
  int argc = 2;
  test_begin_1_xyz(argc, argv);
  use_data = usage;
  tinker_gpu_runtime_initialize();

  const double eps_e = 0.0001;
  const double ref_e = -9.5128;
  const int ref_count = 3;
  const double eps_g = 0.0001;
  const double eps_v = 0.001;
  const double ref_v[][3] = {
      {0.384, -0.444, 0.347}, {-0.444, 1.642, 0.528}, {0.347, 0.528, -2.027}};

  COMPARE_BONDED_FORCE(etortor, ett, ref_e, eps_e, ntortor, ref_count, gx, gy,
                       gz, ref_g_tortor_trpcage, eps_g, vir_ett, ref_v, eps_v);

  tinker_gpu_runtime_finish();
  test_end();
}
