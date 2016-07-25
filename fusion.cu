/*
Copyright (c) 2011-2013 Gerhard Reitmayr, TU Graz

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include "fusion.h"

#undef isnan
#undef isfinite

#include <iostream>

#include "stdio.h"

#include <TooN/TooN.h>
#include <TooN/se3.h>
#include <TooN/GR_SVD.h>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SPQRSupport>

#define INVALID -2   // this is used to mark invalid entries in normal or vertex maps

using namespace std;

//__device__ float nn_d[NN_NUM];
//__device__ int nn_i[NN_NUM];

__global__ void initVolume( Volume volume, const float2 val ){
    uint3 pos = make_uint3(thr2pos2());
    for(pos.z = 0; pos.z < volume.size.z; ++pos.z)
    {
        volume.set(pos, val);
        float nn_d[NN_NUM];
        int nn_i[NN_NUM];
        for(int i=0;i<NN_NUM;i++)
        {
            nn_d[i] = 1;
            nn_i[i] = 0;
        }
        volume.set_knn(pos, nn_d, nn_i);
    }
}

__forceinline__ __device__ float sq( const float x ){
    return x*x;
}


//inVol -> outVol
//outVol.data = inVol.data
//outVol(pos * dq / invpose).data = inVol(pos).data
__global__ void warpVolume( Volume inVol, Volume outVol, const Matrix4 pose, GraphNode * graphNodes, int NodeNum)
{
    uint3 pix = make_uint3(thr2pos2());
    for(pix.z = 0; pix.z < inVol.size.z; ++pix.z)
    {
        float nn_d[NN_NUM];
        int nn_i[NN_NUM];
        inVol.get_knn(pix, nn_d, nn_i);
        float temp_d = 0;
        double w[NN_NUM];
        for(int i=0;i<NN_NUM;i++)
        {
            w[i] = __expf(-sq(nn_d[i]) / (10*sq(graphNodes[nn_i[i]].g_w) ) );
            temp_d += w[i];
        }
        Tbx::Dual_quat_cu dq;
        dq = graphNodes[nn_i[0]].g_dq * (w[0] / temp_d);
        for(int i=1;i<NN_NUM;i++)
        {
            dq = dq + graphNodes[nn_i[i]].g_dq * (w[i] / temp_d);
        }
        Tbx::Point3 t_pos = dq.transform( Tbx::Point3(inVol.pos(pix).x, inVol.pos(pix).y, inVol.pos(pix).z));
        float3 projectedVertex = pose * make_float3(t_pos.x, t_pos.y, t_pos.z);
        int3 temp_pix = min(make_int3(outVol.size.x), max(make_int3(0), make_int3(floorf(make_float3((projectedVertex.x * inVol.size.x / inVol.dim.x) - 0.5f, (projectedVertex.y * inVol.size.y / inVol.dim.y) - 0.5f, (projectedVertex.z * inVol.size.z / inVol.dim.z) - 0.5f))) ));
        uint3 o_pix = make_uint3(temp_pix.x, temp_pix.y, temp_pix.z);
        outVol.set(o_pix, inVol[pix]);
        outVol.set_knn(o_pix, nn_d, nn_i);
    }
}

__global__ void raycast( Image<float3> pos3D, Image<float3> normal, const Volume volume, const Matrix4 view, const float nearPlane, const float farPlane, const float step, const float largestep){
    const uint2 pos = thr2pos2();

    const float4 hit = raycast( volume, pos, view, nearPlane, farPlane, step, largestep );
    if(hit.w > 0){
        pos3D[pos] = make_float3(hit);
        float3 surfNorm = volume.grad(make_float3(hit));
        if(length(surfNorm) == 0){
            normal[pos].x = INVALID;
        } else {
            normal[pos] = normalize(surfNorm);
        }
    } else {
        pos3D[pos] = make_float3(0);
        normal[pos] = make_float3(INVALID, 0, 0);
    }
}

__global__ void integrate( Volume vol, const Image<float> depth, const Matrix4 K, const float mu, const float maxweight, GraphNode * graphNodes, int NodeNum){
    uint3 pix = make_uint3(thr2pos2());

    for(pix.z = 0; pix.z < vol.size.z; ++pix.z/*, pos += delta, cameraX += cameraDelta*/){
        Tbx::Dual_quat_cu dq;
        if(NodeNum > NN_NUM)
        {
            float nn_d[NN_NUM];
            int nn_i[NN_NUM];
            vol.get_knn(pix, nn_d, nn_i);
            float temp_d = 0;
            double w[NN_NUM];
            for(int i=0;i<NN_NUM;i++)
            {
                w[i] = __expf(-sq(nn_d[i]) / (10*sq(graphNodes[nn_i[i]].g_w) ) );
                temp_d += w[i];
            }
            dq = graphNodes[nn_i[0]].g_dq * (w[0] / temp_d);
            for(int i=1;i<NN_NUM;i++)
            {
                dq = dq + graphNodes[nn_i[i]].g_dq * (w[i] / temp_d);
            }
        }
        else
        {
            dq = graphNodes[0].g_dq;
        }
//        if(pix.x == 50 && pix.y == 50 && pix.z == 50)
//        {
//            printf("first: %f %f %f %f \n", dq.get_dual_part().coeff[0], dq.get_dual_part().coeff[1], dq.get_dual_part().coeff[2], dq.get_dual_part().coeff[3]);
//            printf("second: %f %f %f %f \n", dq.get_non_dual_part().coeff[0], dq.get_non_dual_part().coeff[1], dq.get_non_dual_part().coeff[2], dq.get_non_dual_part().coeff[3]);
//        }
        Tbx::Point3 t_pos = dq.transform( Tbx::Point3(vol.pos(pix).x, vol.pos(pix).y, vol.pos(pix).z));
        float3 pos = make_float3(t_pos.x, t_pos.y, t_pos.z);
        float3 cameraX = K * pos;
       if(pos.z < 0.0001f) // some near plane constraint
            continue;
        const float2 pixel = make_float2(cameraX.x/cameraX.z + 0.5f, cameraX.y/cameraX.z + 0.5f);
        if(pixel.x < 0 || pixel.x > depth.size.x-1 || pixel.y < 0 || pixel.y > depth.size.y-1)
            continue;
        const uint2 px = make_uint2(pixel.x, pixel.y);
        if(depth[px] == 0)
            continue;
        const float diff = (depth[px] - cameraX.z) * sqrt(1+sq(pos.x/pos.z) + sq(pos.y/pos.z));
        if(diff > -mu){
            const float sdf = fminf(1.f, diff/mu);
            float2 data = vol[pix];
            data.x = clamp((data.y*data.x + sdf)/(data.y + 1), -1.f, 1.f);
            data.y = fminf(data.y+1, maxweight);
            vol.set(pix, data);
        }
    }
}

__global__ void track( Image<TrackData> output, const Image<float3> inVertex, const Image<float3> inNormal , const Image<float3> refVertex, const Image<float3> refNormal, const Matrix4 Ttrack, const Matrix4 view, const float dist_threshold, const float normal_threshold, Volume volume ) {
    const uint2 pixel = thr2pos2();
    if(pixel.x >= inVertex.size.x || pixel.y >= inVertex.size.y )
        return;

    TrackData & row = output[pixel];

    if(inNormal[pixel].x == INVALID ){
        row.result = -1;
        return;
    }

    const float3 projectedVertex = Ttrack * inVertex[pixel];
    const float3 projectedPos = view * projectedVertex;
    const float2 projPixel = make_float2( projectedPos.x / projectedPos.z + 0.5f, projectedPos.y / projectedPos.z + 0.5f);

    int3 temp_pix = min(make_int3(volume.size.x), max(make_int3(0), make_int3(floorf(make_float3((projectedVertex.x * volume.size.x / volume.dim.x) - 0.5f, (projectedVertex.y * volume.size.y / volume.dim.y) - 0.5f, (projectedVertex.z * volume.size.z / volume.dim.z) - 0.5f))) ));
    uint3 vol_pix = make_uint3(temp_pix.x, temp_pix.y, temp_pix.z);
    float nn_d[NN_NUM];
    int nn_i[NN_NUM];
    volume.get_knn(vol_pix, nn_d, nn_i);

    if(projPixel.x < 0 || projPixel.x > refVertex.size.x-1 || projPixel.y < 0 || projPixel.y > refVertex.size.y-1 ){
        row.result = -2;
        return;
    }

    const uint2 refPixel = make_uint2(projPixel.x, projPixel.y);
    const float3 referenceNormal = refNormal[refPixel];

    if(referenceNormal.x == INVALID){
        row.result = -3;
        return;
    }

    const float3 diff = refVertex[refPixel] - projectedVertex;
    const float3 projectedNormal = rotate(Ttrack, inNormal[pixel]);

    if(length(diff) > dist_threshold ){
        row.result = -4;
        return;
    }
    if(dot(projectedNormal, referenceNormal) < normal_threshold){
        row.result = -5;
        return;
    }

    row.result = 1;
    row.error = dot(referenceNormal, diff);
    float pre_J[6];
    ((float3 *)pre_J)[0] = referenceNormal;
    ((float3 *)pre_J)[1] = cross(projectedVertex, referenceNormal);

    double * jtj = row.J + 7;
    // Error part
    row.J[0] = row.error * row.error;

    // JTe part
    for(int i = 0; i < 6; ++i)
        row.J[i+1] = row.error * pre_J[i];

    // JTJ part, unfortunatly the double loop is not unrolled well...
    jtj[0] = pre_J[0] * pre_J[0];
    jtj[1] = pre_J[0] * pre_J[1];
    jtj[2] = pre_J[0] * pre_J[2];
    jtj[3] = pre_J[0] * pre_J[3];
    jtj[4] = pre_J[0] * pre_J[4];
    jtj[5] = pre_J[0] * pre_J[5];

    jtj[6] = pre_J[1] * pre_J[0];
    jtj[7] = pre_J[1] * pre_J[1];
    jtj[8] = pre_J[1] * pre_J[2];
    jtj[9] = pre_J[1] * pre_J[3];
    jtj[10] = pre_J[1] * pre_J[4];
   jtj[11] = pre_J[1] * pre_J[5];

   jtj[12] = pre_J[2] * pre_J[0];
   jtj[13] = pre_J[2] * pre_J[1];
   jtj[14] = pre_J[2] * pre_J[2];
   jtj[15] = pre_J[2] * pre_J[3];
   jtj[16] = pre_J[2] * pre_J[4];
   jtj[17] = pre_J[2] * pre_J[5];

   jtj[18] = pre_J[3] * pre_J[0];
   jtj[19] = pre_J[3] * pre_J[1];
   jtj[20] = pre_J[3] * pre_J[2];
   jtj[21] = pre_J[3] * pre_J[3];
   jtj[22] = pre_J[3] * pre_J[4];
   jtj[23] = pre_J[3] * pre_J[5];

   jtj[24] = pre_J[4] * pre_J[0];
   jtj[25] = pre_J[4] * pre_J[1];
   jtj[26] = pre_J[4] * pre_J[2];
   jtj[27] = pre_J[4] * pre_J[3];
   jtj[28] = pre_J[4] * pre_J[4];
   jtj[29] = pre_J[4] * pre_J[5];

   jtj[30] = pre_J[5] * pre_J[0];
   jtj[31] = pre_J[5] * pre_J[1];
   jtj[32] = pre_J[5] * pre_J[2];
   jtj[33] = pre_J[5] * pre_J[3];
   jtj[34] = pre_J[5] * pre_J[4];
   jtj[35] = pre_J[5] * pre_J[5];

   //node index
   for(int i=0;i<NN_NUM;i++)
   {
       row.J[i+43] = (double)nn_i[i];
//       printf(" %f", (double)nn_i[i]);
   }

//   printf(" %f %f %f %f", row.J[43], row.J[44], row.J[45], row.J[46]);
}

__global__ void depth2vertex( Image<float3> vertex, const Image<float> depth, const Matrix4 invK ){
    const uint2 pixel = thr2pos2();
    if(pixel.x >= depth.size.x || pixel.y >= depth.size.y )
        return;

    if(depth[pixel] > 0){
        vertex[pixel] = depth[pixel] * (rotate(invK, make_float3(pixel.x, pixel.y, 1.f)));
    } else {
        vertex[pixel] = make_float3(0);
    }
}

__global__ void vertex2knnMap(Image<int> k_map, Image<float3> vertex, Volume vol){
    const uint2 pixel = thr2pos2();
    if(pixel.x >= vertex.size.x || pixel.y >= vertex.size.y )
        return;

    const float3 projectedVertex = vertex[pixel];
    int3 v_pos = min(make_int3(vol.size.x), max(make_int3(0), make_int3(floorf(make_float3((projectedVertex.x * vol.size.x / vol.dim.x) - 0.5f, (projectedVertex.y * vol.size.y / vol.dim.y) - 0.5f, (projectedVertex.z * vol.size.z / vol.dim.z) - 0.5f))) ));
    float nn_d[NN_NUM];
    int nn_i[NN_NUM];
    vol.get_knn(make_uint3(v_pos.x, v_pos.y, v_pos.z), nn_d, nn_i);

    if(nn_d[0] > 0.25)
    {
        k_map[pixel] = 1;
    }
    else
    {
        k_map[pixel] = 0;
    }
}

__global__ void vertex2normal( Image<float3> normal, const Image<float3> vertex ){
    const uint2 pixel = thr2pos2();
    if(pixel.x >= vertex.size.x || pixel.y >= vertex.size.y )
        return;

    const float3 left = vertex[make_uint2(max(int(pixel.x)-1,0), pixel.y)];
    const float3 right = vertex[make_uint2(min(pixel.x+1,vertex.size.x-1), pixel.y)];
    const float3 up = vertex[make_uint2(pixel.x, max(int(pixel.y)-1,0))];
    const float3 down = vertex[make_uint2(pixel.x, min(pixel.y+1,vertex.size.y-1))];

    if(left.z == 0 || right.z == 0 || up.z == 0 || down.z == 0) {
         normal[pixel].x = INVALID;
         return;
    }

    const float3 dxv = right - left;
    const float3 dyv = down - up;
    normal[pixel] = normalize(cross(dyv, dxv)); // switched dx and dy to get factor -1
}

template <int HALFSAMPLE>
__global__ void mm2meters( Image<float> depth, const Image<ushort> in ){
    const uint2 pixel = thr2pos2();
    depth[pixel] = in[pixel * (HALFSAMPLE+1)] / 1000.0f;
}

//column pass using coalesced global memory reads
__global__ void bilateral_filter(Image<float> out, const Image<float> in, const Image<float> gaussian, const float e_d, const int r) {
    const uint2 pos = thr2pos2();

    if(in[pos] == 0){
        out[pos] = 0;
        return;
    }

    float sum = 0.0f;
    float t = 0.0f;
    const float center = in[pos];

    for(int i = -r; i <= r; ++i) {
        for(int j = -r; j <= r; ++j) {
            const float curPix = in[make_uint2(clamp(pos.x + i, 0u, in.size.x-1), clamp(pos.y + j, 0u, in.size.y-1))];
            if(curPix > 0){
                const float mod = sq(curPix - center);
                const float factor = gaussian[make_uint2(i + r, 0)] * gaussian[make_uint2(j + r, 0)] * __expf(-mod / (2 * e_d * e_d));
                t += factor * curPix;
                sum += factor;
            }
        }
    }
    out[pos] = t / sum;
}

// filter and halfsample
__global__ void halfSampleRobust( Image<float> out, const Image<float> in, const float e_d, const int r){
    const uint2 pixel = thr2pos2();
    const uint2 centerPixel = 2 * pixel;

    if(pixel.x >= out.size.x || pixel.y >= out.size.y )
        return;

    float sum = 0.0f;
    float t = 0.0f;
    const float center = in[centerPixel];
    for(int i = -r + 1; i <= r; ++i){
        for(int j = -r + 1; j <= r; ++j){
            float current = in[make_uint2(clamp(make_int2(centerPixel.x + j, centerPixel.y + i), make_int2(0), make_int2(in.size.x - 1, in.size.y - 1)))]; // TODO simplify this!
            if(fabsf(current - center) < e_d){
                sum += 1.0f;
                t += current;
            }
        }
    }
    out[pixel] = t / sum;
}

__global__ void generate_gaussian(Image<float> out, float delta, int radius) {
    int x = threadIdx.x - radius;
    out[make_uint2(threadIdx.x,0)] = __expf(-(x * x) / (2 * delta * delta));
}

void KFusion::Init( const KFusionConfig & config ) {
    configuration = config;

    cudaSetDeviceFlags(cudaDeviceMapHost);

    integration.init(config.volumeSize, config.volumeDimensions);
    warp_vol.init(config.volumeSize, config.volumeDimensions);

    reduction.alloc(config.inputSize);
    vertex.alloc(config.inputSize);
    vertex_pose.alloc(config.inputSize);
    normal.alloc(config.inputSize);
    normal_pose.alloc(config.inputSize);
    rawDepth.alloc(config.inputSize);
    knn_map.alloc(config.inputSize);

    inputDepth.resize(config.iterations.size());
    inputVertex.resize(config.iterations.size());
    inputNormal.resize(config.iterations.size());

    for(int i = 0; i < config.iterations.size(); ++i){
        inputDepth[i].alloc(config.inputSize >> i);
        inputVertex[i].alloc(config.inputSize >> i);
        inputNormal[i].alloc(config.inputSize >> i);
    }

    gaussian.alloc(make_uint2(config.radius * 2 + 1, 1));
    output.alloc(make_uint2(32,8));

    //generate gaussian array
    generate_gaussian<<< 1, gaussian.size.x>>>(gaussian, config.delta, config.radius);

    Reset();
}

void KFusion::Reset(){
    dim3 block(32,16);
    dim3 grid = divup(dim3(integration.size.x, integration.size.y), block);
    initVolume<<<grid, block>>>(integration, make_float2(1.0f, 0.0f));
 }

void KFusion::Clear(){
    integration.release();
}

void KFusion::setPose( const Matrix4 & p ){
    pose = p;
    init_pose = p;
}

void KFusion::setKinectDeviceDepth( const Image<uint16_t> & in){
    if(configuration.inputSize.x == in.size.x)
        mm2meters<0><<<divup(rawDepth.size, configuration.imageBlock), configuration.imageBlock>>>(rawDepth, in);
    else if(configuration.inputSize.x == in.size.x / 2 )
        mm2meters<1><<<divup(rawDepth.size, configuration.imageBlock), configuration.imageBlock>>>(rawDepth, in);
    else
        assert(false);
}

Matrix4 operator*( const Matrix4 & A, const Matrix4 & B){
    Matrix4 R;
    TooN::wrapMatrix<4,4>(&R.data[0].x) = TooN::wrapMatrix<4,4>(&A.data[0].x) * TooN::wrapMatrix<4,4>(&B.data[0].x);
    return R;
}

template<typename P>
inline Matrix4 toMatrix4( const TooN::SE3<P> & p){
    const TooN::Matrix<4, 4, float> I = TooN::Identity;
    Matrix4 R;
    TooN::wrapMatrix<4,4>(&R.data[0].x) = p * I;
    return R;
}

Matrix4 inverse( const Matrix4 & A ){
    static TooN::Matrix<4, 4, float> I = TooN::Identity;
    TooN::Matrix<4,4,float> temp =  TooN::wrapMatrix<4,4>(&A.data[0].x);
    Matrix4 R;
    TooN::wrapMatrix<4,4>(&R.data[0].x) = TooN::gaussian_elimination(temp , I );
    return R;
}

//void KFusion::updateGraphNodes(vector<GraphNode> & graphNodes, Eigen::VectorXd & delta, int idx)
//{
//    TooN::Vector<6> x;
//    x[0] = delta[0]; x[1] = delta[1]; x[2] = delta[2]; x[3] = delta[3]; x[4] = delta[4]; x[5] = delta[5];
//    TooN::SE3<> resultI(x);
//    Matrix4 pose_delta = toMatrix4( resultI );
////    Matrix4 pose_old = pose;
////    pose = pose_delta * pose_old;
//    Matrix4 invPose = inverse(pose_delta);
//    Tbx::Transfo trans(invPose.data[0].x, invPose.data[0].y, invPose.data[0].z, invPose.data[0].w,
//                        invPose.data[1].x, invPose.data[1].y, invPose.data[1].z, invPose.data[1].w,
//                        invPose.data[2].x, invPose.data[2].y, invPose.data[2].z, invPose.data[2].w,
//                        invPose.data[3].x, invPose.data[3].y, invPose.data[3].z, invPose.data[3].w);

////    for(int i=0;i<graphNodes.size();i++)
////    {
//        Tbx::Dual_quat_cu dq = graphNodes[idx].g_dq;
//        Tbx::Transfo trans_orign = dq.to_transformation();
//        trans_orign *= trans;
////        cout<<trans_orign.m[0]<<" "<<trans_orign.m[1]<<" "<<trans_orign.m[2]<<" "<<trans_orign.m[3]<<" "<<endl
////                                   <<trans_orign.m[4]<<" "<<trans_orign.m[5]<<" "<<trans_orign.m[6]<<" "<<trans_orign.m[7]<<" "<<endl
////                                  <<trans_orign.m[8]<<" "<<trans_orign.m[9]<<" "<<trans_orign.m[10]<<" "<<trans_orign.m[11]<<" "<<endl
////                                 <<trans_orign.m[12]<<" "<<trans_orign.m[13]<<" "<<trans_orign.m[14]<<" "<<trans_orign.m[15]<<endl<<endl;
//        graphNodes[idx].g_dq = Tbx::Dual_quat_cu(trans_orign);
////    }
//}
void updateGraphNodes(vector<GraphNode> & graphNodes, Eigen::VectorXd & delta)
{
    for(int i=0;i<graphNodes.size();i++)
    {
        TooN::Vector<6> x;
        x[0] = delta[6*i]; x[1] = delta[6*i+1]; x[2] = delta[6*i+2]; x[3] = delta[6*i+3]; x[4] = delta[6*i+4]; x[5] = delta[6*i+5];
        TooN::SE3<> resultI(x);
        Matrix4 p_delta = toMatrix4( resultI );
        Matrix4 pose_delta = inverse(p_delta);
        Tbx::Transfo trans(pose_delta.data[0].x, pose_delta.data[0].y, pose_delta.data[0].z, pose_delta.data[0].w,
                            pose_delta.data[1].x, pose_delta.data[1].y, pose_delta.data[1].z, pose_delta.data[1].w,
                            pose_delta.data[2].x, pose_delta.data[2].y, pose_delta.data[2].z, pose_delta.data[2].w,
                            pose_delta.data[3].x, pose_delta.data[3].y, pose_delta.data[3].z, pose_delta.data[3].w);
        Tbx::Dual_quat_cu dq = graphNodes[i].g_dq;
        Tbx::Transfo trans_orign = dq.to_transformation();
        trans_orign *= trans;
        graphNodes[i].g_dq = Tbx::Dual_quat_cu(trans_orign);
    }
}

TooN::Vector<6> solve(Eigen::Matrix<double,6,6> & m_J, const Eigen::VectorXd & residual){
    TooN::Vector<6> b;// = TooN::wrapVector<6>(&residual);
    TooN::Matrix<6> C;// = TooN::wrapMatrix<6,6>(&m_J);
    for(int i=0;i<6;i++)
    {
        for(int j=0;j<6;j++)
        {
            C[i][j] = m_J(i, j);
        }
        b[i] = residual[i];
    }

    TooN::GR_SVD<6,6> svd(C);
    return svd.backsub(b, 1e6);
}

void KFusion::Raycast(){
    // raycast integration volume into the depth, vertex, normal buffers
    raycastPose = pose;
    raycast<<<divup(configuration.inputSize, configuration.raycastBlock), configuration.raycastBlock>>>(vertex.getDeviceImage(), normal, integration, raycastPose * getInverseCameraMatrix(configuration.camera), configuration.nearPlane, configuration.farPlane, configuration.stepSize(), 0.75f * configuration.mu);
}

void KFusion::Raycast_pose(vector<GraphNode> & graphNodes){
    // raycast integration volume into the depth, vertex, normal buffers

    thrust::device_vector<GraphNode>  d_graphNodes = graphNodes;
    warpVolume<<<divup(dim3(integration.size.x, integration.size.y), configuration.imageBlock), configuration.imageBlock>>>(integration, warp_vol, init_pose, thrust::raw_pointer_cast(&d_graphNodes[0]), d_graphNodes.size());

    raycastPose = init_pose;
    raycast<<<divup(configuration.inputSize, configuration.raycastBlock), configuration.raycastBlock>>>(vertex_pose.getDeviceImage(), normal_pose, warp_vol, raycastPose * getInverseCameraMatrix(configuration.camera), configuration.nearPlane, configuration.farPlane, configuration.stepSize(), 0.75f * configuration.mu);
}

void KFusion::Prepare() {
    vector<dim3> grids;
    for(int i = 0; i < configuration.iterations.size(); ++i)
        grids.push_back(divup(configuration.inputSize >> i, configuration.imageBlock));

    // filter the input depth map
    bilateral_filter<<<grids[0], configuration.imageBlock>>>(inputDepth[0], rawDepth, gaussian, configuration.e_delta, configuration.radius);

    // half sample the input depth maps into the pyramid levels
    for(int i = 1; i < configuration.iterations.size(); ++i)
        halfSampleRobust<<<grids[i], configuration.imageBlock>>>(inputDepth[i], inputDepth[i-1], configuration.e_delta * 3, 1);

    // prepare the 3D information from the input depth maps
    for(int i = 0; i < configuration.iterations.size(); ++i){
        depth2vertex<<<grids[i], configuration.imageBlock>>>( inputVertex[i], inputDepth[i], getInverseCameraMatrix(configuration.camera / float(1 << i))); // inverse camera matrix depends on level
        vertex2normal<<<grids[i], configuration.imageBlock>>>( inputNormal[i], inputVertex[i]);
    }
}

void KFusion::FindUnmap() {
    vector<dim3> grids;
    for(int i = 0; i < configuration.iterations.size(); ++i)
        grids.push_back(divup(configuration.inputSize >> i, configuration.imageBlock));

    vertex2knnMap<<<grids[0], configuration.imageBlock>>>( knn_map.getDeviceImage(), vertex.getDeviceImage(), integration);
}

vector <Eigen::Matrix<double, 6,6> > m_J(1000);
vector <Eigen::Matrix<double, 1,6> > m_E(1000);

int  KFusion::constructAndSolveJacobi(TrackData *inJacobi, vector<GraphNode> & graphNodes, const bool firstRun)
{
    int width = configuration.inputSize.x;
    int height = configuration.inputSize.y;
    Eigen::Matrix<double, 6,6> tempJ;
    Eigen::Matrix<double, 1,6> tempE;
    double t_data[36];
    for(int i=0;i<36;i++)
    {
        t_data[i] = 0;
    }
    for(int i=0;i<m_J.size();i++)
    {
        memcpy(&m_J[i], t_data, 36*sizeof(double));
        memcpy(&m_E[i], t_data, 6*sizeof(double));
    }
    for(int i=0;i<width;i++)
    {
        for(int j=0;j<height;j++)
        {
            TrackData & rowJ = inJacobi[i + j * width];
            if(rowJ.result == 1)
            {
                //JTJ
                memcpy(&tempJ, &rowJ.J[7], 36*sizeof(double));
                //JTE
                memcpy(&tempE, &rowJ.J[1], 6*sizeof(double));
                for(int nn=0;nn<NN_NUM;nn++)
                {
//                    cout<<rowJ.J[nn + 43]<<" ";
                    int i1 = (int)rowJ.J[nn + 43];
                    if(i1<1000 && i1 > 0)
                    {
                        m_J[i1] +=  tempJ;
                        m_E[i1] +=  tempE;
//                        m_J[0] +=  tempJ;
//                        m_E[0] +=  tempE;
                    }
                }
            }
        }
    }

    //find positive definite
    int ii = 0;
//    int idx[1000];
    for(int i=0;i<graphNodes.size();i++)
    {
        Eigen::EigenSolver<Eigen::Matrix<double,6,6> > eigensolver(m_J[i]);
        if(eigensolver.eigenvalues()[0].imag() == 0 && eigensolver.eigenvalues()[0].real() > 0)
        {
//            idx[ii] = i;
            ii++;
        }
    }

    if(ii > 0)
    {
        Eigen::SparseMatrix < double > jtj ( 6000 , 6000 ) ;
        Eigen::VectorXd jte(6000);
        std :: vector < Eigen::Triplet < double > > triplets ;

        for(int i=0;i<6000;i++)
        {
            jte[i] = 0;
        }

        for(int i=0;i<graphNodes.size();i++)
        {
            int *reg_ind = graphNodes[i].reg_ind;
            for(int r=0;r<6;r++)
            {
                for(int c=0;c<6;c++)
                {
                    double t_data = m_J[i](r,c);
                    double t_reg = m_J[reg_ind[0]](r,c) + m_J[reg_ind[1]](r,c) + m_J[reg_ind[2]](r,c) + m_J[reg_ind[3]](r,c);//Regularization
                    double t_val = t_data+0.05*t_reg;
                    if(t_val != 0){
                        triplets.push_back(Eigen::Triplet <double> (6*i+r, 6*i+c, t_val));
                         //Regularization
                        triplets.push_back(Eigen::Triplet <double> (6*i+r, 6*reg_ind[0]+c, 0.05*m_J[reg_ind[0]](r,c)));
                        triplets.push_back(Eigen::Triplet <double> (6*i+r, 6*reg_ind[1]+c, 0.05*m_J[reg_ind[1]](r,c)));
                        triplets.push_back(Eigen::Triplet <double> (6*i+r, 6*reg_ind[2]+c, 0.05*m_J[reg_ind[2]](r,c)));
                        triplets.push_back(Eigen::Triplet <double> (6*i+r, 6*reg_ind[3]+c, 0.05*m_J[reg_ind[3]](r,c)));
                    }
                }
                jte[i*6 + r] = m_E[i][r];
            }
        }

        // 初始化稀疏矩阵
        jtj.setFromTriplets ( triplets.begin() , triplets.end() ) ;

        Eigen::SPQR < Eigen::SparseMatrix < double > > qr_jtj ;
        qr_jtj.compute(jtj);
        Eigen::VectorXd delta = qr_jtj.solve(jte);
        updateGraphNodes(graphNodes, delta);
    }

    return 0;
}

bool KFusion::Track(vector<GraphNode> & graphNodes) {
    vector<dim3> grids;
    for(int i = 0; i < configuration.iterations.size(); ++i)
        grids.push_back(divup(configuration.inputSize >> i, configuration.imageBlock));

//    const Matrix4 oldPose = pose;
    const Matrix4 projectReference = getCameraMatrix(configuration.camera) * inverse(init_pose);

//    thrust::device_vector<GraphNode>  d_graphNodes = graphNodes;

    if(graphNodes.size() > 10)
    {
        for(int i=0;i<2;i++)
        {
            track<<<grids[0], configuration.imageBlock>>>( reduction.getDeviceImage(), inputVertex[0], inputNormal[0], vertex_pose.getDeviceImage(), normal_pose, init_pose, projectReference, configuration.dist_threshold, configuration.normal_threshold, warp_vol/*, thrust::raw_pointer_cast(&d_graphNodes[0]), d_graphNodes.size()*/ );
            cudaDeviceSynchronize(); // important due to async nature of kernel call
            constructAndSolveJacobi(reduction.data(), graphNodes, 1);
            Integrate(graphNodes);
            Raycast_pose(graphNodes);
        }
    }

    return true;
}

void KFusion::Integrate(vector<GraphNode> & graphNodes) {

    thrust::device_vector<GraphNode>  d_graphNodes = graphNodes;
    integrate<<<divup(dim3(integration.size.x, integration.size.y), configuration.imageBlock), configuration.imageBlock>>>( integration, rawDepth,  getCameraMatrix(configuration.camera), configuration.mu, configuration.maxweight, thrust::raw_pointer_cast(&d_graphNodes[0]), d_graphNodes.size() );
}

int printCUDAError() {
    cudaError_t error = cudaGetLastError();
    if(error)
        std::cout << cudaGetErrorString(error) << std::endl;
    return error;
}
//-----------------------------------------------------------------------------------------------//
//                                            KNN KERNELS                                            //
//-----------------------------------------------------------------------------------------------//

/**
  * Computes the distance
  */
__global__ void cuComputeDistanceNodes( GraphNode * graphNodes, Volume volume){

    uint3 pos = make_uint3(thr2pos2());
    for(pos.z = 0; pos.z < volume.size.z; ++pos.z)
    {
        float3 m_pos = volume.pos(pos);
        float temp_d[NN_NUM];
        int temp_i[NN_NUM];
        volume.get_knn(pos,temp_d, temp_i );
        for (int i = 0; i < NN_NUM; ++i){
            temp_d[i] = (m_pos.x - graphNodes[i].g_v.x) * (m_pos.x - graphNodes[i].g_v.x)
                + (m_pos.y - graphNodes[i].g_v.y) * (m_pos.y - graphNodes[i].g_v.y)
                + (m_pos.z - graphNodes[i].g_v.z) * (m_pos.z - graphNodes[i].g_v.z);
            temp_i[i] = i;
          }
          volume.set_knn(pos, temp_d, temp_i);
    }
//    printf(" %d",volume.size.z);
}

/**
  * Gathers k-th smallest distances for each column.
  */
__global__ void cuInsertionSortNodes(GraphNode * graphNodes, Volume volume, int noTotalNodes){

  // Variables
  int current_ind, i, j;
  int k = NN_NUM;
  float p_dist[NN_NUM];
  int   p_ind[NN_NUM];
  float curr_dist, max_dist;
  int   max_row;

  uint3 pos = make_uint3(thr2pos2());
  for(pos.z = 0; pos.z < volume.size.z; ++pos.z)
  {
      float nn_d[NN_NUM];
      int nn_i[NN_NUM];
      volume.get_knn(pos, nn_d, nn_i);
      for(int t=0;t<NN_NUM;t++)
      {
          p_dist[t] = nn_d[t];
      }
      max_dist = nn_d[0];
      p_ind[0] = 0;

      // Part 1 : sort kth first elementZ
      for (current_ind=1; current_ind<k; current_ind++)
      {
          curr_dist = p_dist[current_ind];
          if (curr_dist<max_dist)
          {
              i=current_ind-1;
              for (int a=0; a<current_ind-1; a++)
              {
                  if (p_dist[a]>curr_dist)
                  {
                      i=a;
                      break;
                    }
                }
              for (j=current_ind; j>i; j--)
              {
                  p_dist[j] = p_dist[j-1];
                  p_ind[j]   = p_ind[j-1];
                }
              p_dist[i] = curr_dist;
              p_ind[i]   = current_ind+1;
            }
          else
          {
            p_ind[current_ind] = current_ind+1;
          }
          max_dist = p_dist[current_ind];
        }

      // Part 2 : insert element in the k-th first lines
      max_row = k-1;
      for (current_ind=k; current_ind<noTotalNodes; current_ind++)
      {
          float3 curr_pos = volume.pos(pos);
          curr_dist = (curr_pos.x - graphNodes[current_ind].g_v.x) * (curr_pos.x - graphNodes[current_ind].g_v.x)
                  + (curr_pos.y - graphNodes[current_ind].g_v.y) * (curr_pos.y - graphNodes[current_ind].g_v.y)
                  + (curr_pos.z - graphNodes[current_ind].g_v.z) * (curr_pos.z - graphNodes[current_ind].g_v.z);

    //      __syncthreads();

          if (curr_dist<max_dist){
              i=k-1;
              for (int a=0; a<k-1; a++){
                  if (p_dist[a]>curr_dist){
                      i=a;
                      break;
                    }
                }
              for (j=k-1; j>i; j--){
                  p_dist[j] = p_dist[j-1];
                  p_ind[j]   = p_ind[j-1];
                }
              p_dist[i] = curr_dist;
              p_ind[i]  = current_ind+1;
              max_dist  = p_dist[max_row];
            }
        }
      volume.set_knn(pos, p_dist, p_ind);
  }
}

/**
  * Computes the square root of the first line (width-th first element)
  */
__global__ void cuParallelSqrtNodes(Volume volume){
    uint3 pos = make_uint3(thr2pos2());
    for(pos.z = 0; pos.z < volume.size.z; ++pos.z)
    {
        float nn_d[NN_NUM];
        int nn_i[NN_NUM];
        volume.get_knn(pos, nn_d, nn_i);
        for(int i=0;i<NN_NUM;i++)
        {
            nn_d[i] = sqrt(nn_d[i]);
        }
        volume.set_knn(pos, nn_d, nn_i);
    }
}

//__global__
//void Kernel(GraphNode* dv)
//{
//    int i = threadIdx.x;
//    printf("%f\n", dv[1].g_vv.x);
//}

void KFusion::findKnn( vector<GraphNode> & graphNodes)
{
    dim3 block(32,16);
    dim3 grid = divup(dim3(integration.size.x, integration.size.y), block);

    thrust::device_vector<GraphNode>  d_graphNodes = graphNodes;

//    vector<GraphNode> test(2);test[0].g_vv = Tbx::Point3(2,3,4);test[1].g_vv = Tbx::Point3(2,3,4);
////    printf("%f\n", test[1].g_vv.x);
//    thrust::device_vector<GraphNode> bbb = test;

//    Kernel<<<grid,block>>>(thrust::raw_pointer_cast(&bbb[0]));

    // Kernel 1: Compute all the distances
    cuComputeDistanceNodes<<<grid,block>>>(thrust::raw_pointer_cast(&d_graphNodes[0]), integration);

    // Kernel 2: sort
    cuInsertionSortNodes<<<grid,block>>>(thrust::raw_pointer_cast(&d_graphNodes[0]), integration, d_graphNodes.size());

    // Kernel 3: sqrt
    cuParallelSqrtNodes<<<grid,block>>>(integration);
}
