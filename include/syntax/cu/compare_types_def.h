#pragma once
#include "macro.h"
#include <type_traits>


TINKER_NAMESPACE_BEGIN
template <class T, class U>
__device__
__host__
constexpr bool eq()
{
   return std::is_same<T, U>::value;
}
TINKER_NAMESPACE_END