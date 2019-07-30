#include "files.h"
#include "test/ff.h"
#include "test/rt.h"
#include "test/test.h"

using namespace TINKER_NAMESPACE;
using namespace test;

static const char* triclinic_box = R"**(
ewald
ewald-cutoff  7.0
vdw-cutoff    7.0
neighbor-list
list-buffer  0.01

a-axis   28.0
b-axis   22.0
c-axis   18.0
alpha   105.0
beta    110.0
gamma    80.0
)**";

static const char* monoclinic_box = R"**(
ewald
ewald-cutoff  7.0
vdw-cutoff    7.0
neighbor-list
list-buffer  0.01

a-axis   28.0
b-axis   22.0
c-axis   18.0
alpha    90.0
beta    110.0
gamma    90.0
)**";

static int usage = calc::xyz | calc::vmask;
static const char* k = "test_local_frame2.key";
static const char* x1 = "test_local_frame2.xyz";
static const char* argv[] = {"dummy", x1};
static int argc = 2;

TEST_CASE("Local-Frame2-1", "[ff][triclinic][evdw][hal][local-frame2]") {
  std::string k0 = local_frame_key;
  k0 += triclinic_box;
  k0 += "vdwterm  only\n";
  file fke(k, k0);

  file fpr("amoeba09.prm", amoeba09_prm);
  file fx1(x1, local_frame_xyz2);

  test_begin_1_xyz(argc, argv);
  use_data = usage;
  tinker_gpu_runtime_initialize();

  SECTION("ehal -- pbc, cutoff") {
    const double eps_e = 0.0001;
    const double ref_eng = 206.5670;
    const int ref_count = 129;

    zero_egv();
    evdw_hal(calc::v3);
    COMPARE_ENERGY_(ev, ref_eng, eps_e);
    COMPARE_COUNT_(nev, ref_count);
  }

  tinker_gpu_runtime_finish();
  test_end();
}

TEST_CASE("Local-Frame2-2", "[ff][monoclinic][evdw][hal][local-frame2]") {
  std::string k0 = local_frame_key;
  k0 += monoclinic_box;
  k0 += "vdwterm  only\n";
  file fke(k, k0);

  file fpr("amoeba09.prm", amoeba09_prm);
  file fx1(x1, local_frame_xyz2);

  test_begin_1_xyz(argc, argv);
  use_data = usage;
  tinker_gpu_runtime_initialize();

  SECTION("ehal -- pbc, cutoff") {
    const double eps_e = 0.0001;
    const double ref_eng = 182.5400;
    const int ref_count = 125;

    zero_egv();
    evdw_hal(calc::v3);
    COMPARE_ENERGY_(ev, ref_eng, eps_e);
    COMPARE_COUNT_(nev, ref_count);
  }

  tinker_gpu_runtime_finish();
  test_end();
}
