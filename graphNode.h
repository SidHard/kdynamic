#pragma once

#include "thirdparty/dualquaternion/point3.hpp"
#include "thirdparty/dualquaternion/vec3.hpp"
#include "thirdparty/dualquaternion/mat3.hpp"
#include "thirdparty/dualquaternion/dual_quat_cu.hpp"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using namespace std;

struct GraphNode
{
        int id;
//        float g_vv[3];
        Tbx::Point3 g_v;
        float g_w;
        Tbx::Dual_quat_cu g_dq;
        int reg_ind[4];
        float reg_dist[4];
        bool IsReg;
        int reg_ind2[4];
        float reg_dist2[4];
};
