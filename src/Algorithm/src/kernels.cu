#include"kernels.h"
#include"device_code.h"
#include"sMatrix.h"
#include<stdint.h>
#include"constant_parameters.h"
#include"rgb2Lab.h"

// #define USE_LAB

__global__ void getVoxelData(Volume vol, short2 *output)
{
    uint3 pix = make_uint3(thr2pos2());
    for (pix.z = 0; pix.z < vol.getResolution().z; pix.z++)
    {
        int idx= pix.x + pix.y * vol.getResolution().x + pix.z * vol.getResolution().x * vol.getResolution().y;
        float2 p_data = vol[pix];
        output[idx]=make_short2(p_data.x * 32766.0f, p_data.y);
    }
}

__global__ void renderDepthKernel(Image<uchar3> out,
                                  const Image<float> depth,
                                  const float nearPlane,
                                  const float farPlane)
{
    const float d = (clamp(depth.el(), nearPlane, farPlane) - nearPlane) / (farPlane - nearPlane);
    out.el() = make_uchar3(d * 255, d * 255, d * 255);
}

__global__ void renderTrackKernel(Image<uchar3> out,const Image<TrackData> data)
{
    const uint2 pos = thr2pos2();
    switch (data[pos].result)
    {
        case  1: out[pos] = make_uchar3(128, 128, 128);	break; // ok
        case -1: out[pos] = make_uchar3(0, 0, 0);	break; // no input
        case -2: out[pos] = make_uchar3(255, 0, 0);	break; // not in image
        case -3: out[pos] = make_uchar3(0, 255, 0);	break; // no correspondence
        case -4: out[pos] = make_uchar3(0, 0, 255);	break; // too far away
        case -5: out[pos] = make_uchar3(255, 255, 0);	break; // wrong normal
    }
}

__global__ void renderVolumeKernel(Image<uchar3> render,
                                   Image<float3> vertex,
                                   Image<float3> normal,
                                   const float3 light,
                                   const float3 ambient)
{
    const uint2 pos = thr2pos2();
    if(normal[pos].x != INVALID)
    {
        const float3 surfNorm = normal[pos];
        const float3 diff = normalize(light - vertex[pos]);
        const float dir = fmaxf( dot(normalize(surfNorm), diff), 0.f);
        const float3 col = clamp(make_float3(dir) + ambient, 0.f, 1.f)* 255;
        render.el() = make_uchar3(col.x, col.y, col.z);
    }
    else
    {
         render.el() = make_uchar3(0, 0, 0);
    }
}

__global__ void renderVolumeKernel2(Image<uchar3> render,
                                   Image<float3> vertex,
                                   Image<float3> normal,
                                   const float3 light,
                                   const float3 ambient,
                                   const float nearPlane,
                                   const float farPlane)

{
    const uint2 pos = thr2pos2();
    if(normal[pos].x != INVALID)
    {
        const float3 surfNorm = normal[pos];
        const float3 diff =  vertex[pos];
        const float dir = fmaxf( dot(normalize(surfNorm), diff), 0.f);

        const float d = (clamp(dir, nearPlane, farPlane) - nearPlane) / (farPlane - nearPlane);
        render.el() = make_uchar3(d * 255, d * 255, d * 255);
    }
    else
    {
         render.el() = make_uchar3(0, 0, 0);
    }
}

__global__ void vertex2depthKernel(Image<float> render,
                                   Image<float3> vertex,
                                   const sMatrix4 K)

{
    const uint2 pixel = thr2pos2();

    if (pixel.x >= vertex.size.x || pixel.y >= vertex.size.y)
        return;

    float3 v= vertex[pixel];
    float3 tmp=rotate(K,v);

    if(tmp.z<=0.0)
    {
        render.el() = 0.0;
    }
    else
    {
        float depth=tmp.z;
        render.el() = depth;
    }
}

__global__ void initVolumeKernel(Volume volume,const float2 val)
{
    uint3 pos = make_uint3(thr2pos2());
    for (pos.z=0; pos.z < volume.getResolution().z; pos.z++)
    {
        volume.set(pos, val);
    }
}

__global__ void raycastKernel(Image<float3> pos3D,
                              Image<float3> normal,
                              const Volume volume,
                              const sMatrix4 view,
                              const float nearPlane,
                              const float farPlane,
                              const float step,
                              const float largestep,
                              int frame)
{
    const uint2 pos = thr2pos2();
    const float4 hit = raycast(volume, pos, view, nearPlane, farPlane, step,largestep,frame);

    if (hit.w > 0)
    {
        pos3D[pos] = make_float3(hit);
        float3 surfNorm = volume.grad(make_float3(hit));
        if (length(surfNorm) == 0)
        {
            normal[pos].x = INVALID;
        }
        else
        {
            normal[pos] = normalize(surfNorm);
        }
    }
    else
    {
        pos3D[pos] = make_float3(0);
        normal[pos] = make_float3(INVALID, 0, 0);
    }
}

__global__ void fuseVolumesKernel(Volume dstVol,
                                  Volume srcVol,
                                  const sMatrix4 pose,
                                  const float3 origin,
                                  const float maxweight)
{
    uint3 pix = make_uint3(thr2pos2());

    if( pix.x >= dstVol.getResolution().x ||
        pix.y >= dstVol.getResolution().y )
    {
        return;
    }

    float3 vsize=srcVol.getSizeInMeters();



    for (pix.z = 0; pix.z < dstVol.getResolution().z; pix.z++)
    {
        float3 pos=dstVol.pos(pix);

        pos=pose*pos;

        pos.x+=origin.x;
        pos.y+=origin.y;
        pos.z+=origin.z;

        if( pos.x<0 || pos.x >= vsize.x ||
            pos.y<0 || pos.y >= vsize.y ||
            pos.z<0 || pos.z >= vsize.z)
        {
             continue;
        }

        float tsdf=srcVol.interp(pos);

        float3 fcol=srcVol.rgb_interp(pos);
        //float w_interp=srcVol.ww_interp(pos);
        float w_interp=1;

//         if(w_interp < 1.0)
//         {
//             continue;
//         }

        float2 p_data = dstVol[pix];
        float3 p_color = dstVol.getColor(pix);

        if(tsdf == 1.0)
            continue;

        float w=p_data.y;
        float new_w=w+w_interp;

        fcol.x = (w*p_color.x + w_interp*fcol.x ) / new_w;
        fcol.y = (w*p_color.y + w_interp*fcol.y ) / new_w;
        fcol.z = (w*p_color.z + w_interp*fcol.z ) / new_w;

        p_data.x = clamp( (w*p_data.x + w_interp*tsdf) / new_w, -1.f, 1.f);
        p_data.y=fminf(new_w, maxweight);

        dstVol.set(pix,p_data, fcol);

    }
}


__global__ void integrateKernel(Volume vol, const Image<float> depth,
                                const Image<uchar3> rgb,
                                const sMatrix4 invTrack,
                                const sMatrix4 K,
                                const float mu,
                                const float maxweight)
{
    int3 pix = make_int3(thr2pos2i());
    float3 pos = invTrack * vol.pos(pix);
    float3 cameraX = K * pos;
    const float3 delta = rotate(invTrack,make_float3(0, 0, vol.getDimensions().z / vol.getResolution().z));
    const float3 cameraDelta = rotate(K, delta);

    int blockIdx=-1;
    for (pix.z = 0; pix.z < vol.getResolution().z; pix.z++, pos += delta, cameraX +=cameraDelta)
    {
        if (pos.z < 0.0001f) // some near plane constraint
            continue;

        const float2 pixel = make_float2(cameraX.x / cameraX.z + 0.5f,
                                         cameraX.y / cameraX.z + 0.5f);

        if (pixel.x < 0 || pixel.x > depth.size.x - 1 ||
            pixel.y < 0 || pixel.y > depth.size.y - 1)
        {
            continue;
        }

        const uint2 px = make_uint2(pixel.x, pixel.y);

        if (depth[px] == 0)
            continue;
        const float diff = (depth[px] - cameraX.z) *
                           sqrt(1 + sq(pos.x / pos.z) + sq(pos.y / pos.z));

        if (diff > -mu)
        {
            const float sdf = fminf(1.f, diff / mu);

            float2 p_data=make_float2(1.0,0.0);
//            float3 p_color=make_float3(0.0, 0.0, 0.0);
            tsdfvh::Voxel *v=vol.insertVoxel(pix,blockIdx);

            if(blockIdx<0)
                continue;

            p_data.x=v->getTsdf();
            p_data.y=v->getWeight();
            float3 p_color = v->color;


            //float w=fmin(p_data.y,maxweight);
            float w=p_data.y;
            float new_w=w+1;

#ifdef USE_LAB
            float3 fcol=rgb2lab(rgb[px]);
#else
            float3 fcol = make_float3(rgb[px].x,rgb[px].y,rgb[px].z);
#endif
            p_data.x = clamp( (w*p_data.x + sdf) / new_w, -1.f, 1.f);

            fcol.x = (w*p_color.x + fcol.x ) / new_w;
            fcol.y = (w*p_color.y + fcol.y ) / new_w;
            fcol.z = (w*p_color.z + fcol.z ) / new_w;

            /*
            frgb.x=clamp(frgb.x,MIN_L,MAX_L);
            frgb.y=clamp(frgb.y,MIN_A,MAX_A);
            frgb.z=clamp(frgb.z,MIN_B,MAX_B);
            */
            //p_data.y=p_data.y+1;
            p_data.y=fminf(new_w, maxweight);
            //vol.set(pix,p_data, fcol);

            v->setTsdf(p_data.x);
            v->setWeight(p_data.y);
            v->color=fcol;

            //printf("v:%f %f %f %f\n",p_data.x,p_data.y,p_data_n.x,p_data_n.y);
        }
    }
}

//TODO fix me
__global__ void deIntegrateKernel(Volume vol,
                                  const Image<float> depth,
                                  const Image<uchar3> rgb,
                                  const sMatrix4 invTrack,
                                  const sMatrix4 K,
                                  const float mu,
                                  const float maxweight)
{
    uint3 pix = make_uint3(thr2pos2());
    float3 pos = invTrack * vol.pos(pix);
    float3 cameraX = K * pos;
    const float3 delta = rotate(invTrack,make_float3(0, 0, vol.getDimensions().z / vol.getResolution().z));
    const float3 cameraDelta = rotate(K, delta);

    for (pix.z=0; pix.z!=vol.getResolution().z; pix.z++, pos += delta, cameraX +=cameraDelta)
    {
        if (pos.z < 0.0001f) // some near plane constraint
            continue;

        const float2 pixel = make_float2(cameraX.x / cameraX.z + 0.5f,
                                         cameraX.y / cameraX.z + 0.5f);

        if (pixel.x < 0 || pixel.x > depth.size.x - 1 || pixel.y < 0|| pixel.y > depth.size.y - 1)
            continue;

        const uint2 px = make_uint2(pixel.x, pixel.y);

        if (depth[px] == 0)
            continue;

        const float diff = (depth[px] - cameraX.z) *
                           sqrt(1 + sq(pos.x / pos.z) + sq(pos.y / pos.z));

        if (diff > -mu)
        {
            const float sdf = fminf(1.f, diff / mu);
            float2 p_data = vol[pix];

            float3 fcol;
            float w=fmin(p_data.y,maxweight);
            float new_w=w-1;
            //if w is 0 restore initial contitions
            if(new_w==0)
            {
                p_data.x = 1;
                p_data.y = 0;
                fcol = make_float3(0.0,0.0,0.0);
            }
            else
            {
#ifdef USE_LAB
                fcol=rgb2lab(rgb[px]);
#else
                fcol = make_float3(rgb[px].x,rgb[px].y,rgb[px].z);
#endif
                float3 p_color = vol.getColor(pix);

                float w=fmin(p_data.y,maxweight);


                p_data.x = clamp( (w * p_data.x - sdf) / new_w,-1.f,1.f);
                fcol.x = (w * p_color.x - fcol.x ) / new_w;
                fcol.y = (w * p_color.y - fcol.y ) / new_w;
                fcol.z = (w * p_color.z - fcol.z ) / new_w;

                /*
                frgb.x=clamp(frgb.x,MIN_L,MAX_L);
                frgb.y=clamp(frgb.y,MIN_A,MAX_A);
                frgb.z=clamp(frgb.z,MIN_B,MAX_B);
                */
                p_data.y = p_data.y-1;
            }

            vol.set(pix,p_data, fcol);
        }
    }
}

__global__ void compareRgbKernel(const Image<uchar3> image1,
                                 const Image<uchar3> image2,
                                 Image<float>out)
{
    const uint2 pixel = thr2pos2();

    uchar3 pix1=image1[pixel];
    uchar3 pix2=image2[pixel];

    float dist=sqrt((float) sq(pix1.x-pix2.x) +
                    (float) sq(pix1.y-pix2.y) +
                    (float) sq(pix1.z-pix2.z) );

    out[pixel]=dist;
}

__global__ void depth2vertexKernel(Image<float3> vertex,const Image<float> depth, const sMatrix4 invK)
{
    const uint2 pixel = thr2pos2();
    if (pixel.x >= depth.size.x || pixel.y >= depth.size.y)
        return;

    if (depth[pixel] > 0)
    {
        vertex[pixel] = depth[pixel]
                        * (rotate(invK, make_float3(pixel.x, pixel.y, 1.f)));
    }
    else
    {
        vertex[pixel] = make_float3(0);
    }
}

__global__ void vertex2normalKernel(Image<float3> normal,const Image<float3> vertex)
{
    const uint2 pixel = thr2pos2();
    if (pixel.x >= vertex.size.x || pixel.y >= vertex.size.y)
        return;

    const float3 left = vertex[make_uint2(max(int(pixel.x) - 1, 0), pixel.y)];
    const float3 right = vertex[make_uint2(min(pixel.x + 1, vertex.size.x - 1),
                                           pixel.y)];
    const float3 up = vertex[make_uint2(pixel.x, max(int(pixel.y) - 1, 0))];
    const float3 down = vertex[make_uint2(pixel.x,
                                          min(pixel.y + 1, vertex.size.y - 1))];

    if (left.z == 0 || right.z == 0 || up.z == 0 || down.z == 0) {
        normal[pixel].x = INVALID;
        return;
    }

    const float3 dxv = right - left;
    const float3 dyv = down - up;
    normal[pixel] = normalize(cross(dyv, dxv)); // switched dx and dy to get factor -1
}

__global__ void mm2metersKernel( Image<float> depth, const Image<ushort> in )
{
    const uint2 pixel = thr2pos2();
    depth[pixel] = (float) in[pixel] / 1000.0f;
}

//column pass using coalesced global memory reads
__global__ void bilateralFilterKernel(Image<float> out,
                                      const Image<float> in,
                                      const Image<float> gaussian,
                                      const float e_d,
                                      const int r)
{
    const uint2 pos = thr2pos2();

    if (in[pos] == 0)
    {
        out[pos] = 0;
        return;
    }

    float sum = 0.0f;
    float t = 0.0f;
    const float center = in[pos];

    for (int i = -r; i <= r; ++i)
    {
        for (int j = -r; j <= r; ++j)
        {
            const float curPix = in[make_uint2(
                                     clamp(pos.x + i, 0u, in.size.x - 1),
                                     clamp(pos.y + j, 0u, in.size.y - 1))];
            if (curPix > 0)
            {
                const float mod = sq(curPix - center);
                const float factor = gaussian[make_uint2(i + r, 0)]
                                     * gaussian[make_uint2(j + r, 0)]
                                     * __expf(-mod / (2 * e_d * e_d));
                t += factor * curPix;
                sum += factor;
            }
        }
    }
    out[pos] = t / sum;
}

//Render the image using the last raycast
__global__ void renderRgbKernel(Image<uchar3> render,
                                const Volume volume,
                                Image<float3> vert,
                                Image<float3> norm)
{
    const uint2 pos = thr2pos2();

    if(norm[pos].x != INVALID)
    {

        float3 vertex=vert[pos];
        if(!volume.isPointInside(vertex) )
        {
            render.el() = make_uchar3(0, 0, 0);
        }
        else
        {
            float3 fcol = volume.rgb_interp(vertex);

#ifdef USE_LAB
            fcol.x=clamp(flab.x,MIN_L,MAX_L);
            fcol.y=clamp(flab.y,MIN_A,MAX_A);
            fcol.z=clamp(flab.z,MIN_B,MAX_B);
            uchar3 rgb=lab2rgb(flab);
#else
            uchar3 rgb=make_uchar3(fcol.x ,fcol.y ,fcol.z);
#endif
            render.el()=rgb;
        }

    }
    else
    {
        render.el() = make_uchar3(0, 0, 0);
    }
}

// filter and halfsample
__global__ void halfSampleRobustImageKernel(Image<float> out,
                                            const Image<float> in,
                                            const float e_d,
                                            const int r)
{
    const uint2 pixel = thr2pos2();
    const uint2 centerPixel = 2 * pixel;

    if (pixel.x >= out.size.x || pixel.y >= out.size.y)
        return;

    float sum = 0.0f;
    float t = 0.0f;
    const float center = in[centerPixel];
    for (int i = -r + 1; i <= r; ++i)
    {
        for (int j = -r + 1; j <= r; ++j)
        {
            float current = in[make_uint2(
                                clamp(make_int2(centerPixel.x + j, centerPixel.y + i),
                                      make_int2(0),
                                      make_int2(in.size.x - 1, in.size.y - 1)))]; // TODO simplify this!
            if (fabsf(current - center) < e_d)
            {
                sum += 1.0f;
                t += current;
            }
        }
    }
    out[pixel] = t / sum;
}

__global__ void generate_gaussian(Image<float> out, float delta, int radius)
{
    int x = threadIdx.x - radius;
    out[make_uint2(threadIdx.x, 0)] = __expf(-(x * x) / (2 * delta * delta));
}

__global__ void trackKernel(Image<TrackData> output,
                            const Image<float3> inVertex,
                            const Image<float3> inNormal,
                            const Image<float3> refVertex,
                            const Image<float3> refNormal,
                            const sMatrix4 Ttrack,
                            const sMatrix4 view,
                            const float dist_threshold,
                            const float normal_threshold)
{
    const uint2 pixel = thr2pos2();
    if (pixel.x >= inVertex.size.x || pixel.y >= inVertex.size.y)
        return;

    TrackData & row = output[pixel];

    if (inNormal[pixel].x == INVALID)
    {
        row.result = -1;
        return;
    }

    const float3 projectedVertex = Ttrack * inVertex[pixel];
    const float3 projectedPos = view * projectedVertex;
    const float2 projPixel = make_float2(projectedPos.x / projectedPos.z + 0.5f,
                                         projectedPos.y / projectedPos.z + 0.5f);

    if (projPixel.x < 0 || projPixel.x > refVertex.size.x - 1 || projPixel.y < 0
            || projPixel.y > refVertex.size.y - 1)
    {
        row.result = -2;
        return;
    }

    const uint2 refPixel = make_uint2(projPixel.x, projPixel.y);
    const float3 referenceNormal = refNormal[refPixel];

    if (referenceNormal.x == INVALID)
    {
        row.result = -3;
        return;
    }

    const float3 diff = refVertex[refPixel] - projectedVertex;
    const float3 projectedNormal = rotate(Ttrack, inNormal[pixel]);

    if (length(diff) > dist_threshold)
    {
        row.result = -4;
        return;
    }
    if (dot(projectedNormal, referenceNormal) < normal_threshold)
    {
        row.result = -5;
        return;
    }

    row.result = 1;
    row.error = dot(referenceNormal, diff);
    ((float3 *) row.J)[0] = referenceNormal;
    ((float3 *) row.J)[1] = cross(projectedVertex, referenceNormal);
}

__global__ void reduceKernel(float * out, const Image<TrackData> J,const uint2 size)
{
    __shared__ float S[112][32]; // this is for the final accumulation
    const uint sline = threadIdx.x;

    float sums[32];
    float * jtj = sums + 7;
    float * info = sums + 28;

    for (uint i = 0; i < 32; ++i)
        sums[i] = 0;

    for (uint y = blockIdx.x; y < size.y; y += gridDim.x)
    {
        for (uint x = sline; x < size.x; x += blockDim.x)
        {
            const TrackData & row = J[make_uint2(x, y)];
            if (row.result < 1)
            {
                info[1] += row.result == -4 ? 1 : 0;
                info[2] += row.result == -5 ? 1 : 0;
                info[3] += row.result > -4 ? 1 : 0;
                continue;
            }

            // Error part
            sums[0] += row.error * row.error;

            // JTe part
            for (int i = 0; i < 6; ++i)
                sums[i + 1] += row.error * row.J[i];

            // JTJ part, unfortunatly the float loop is not unrolled well...
            jtj[0] += row.J[0] * row.J[0];
            jtj[1] += row.J[0] * row.J[1];
            jtj[2] += row.J[0] * row.J[2];
            jtj[3] += row.J[0] * row.J[3];
            jtj[4] += row.J[0] * row.J[4];
            jtj[5] += row.J[0] * row.J[5];

            jtj[6] += row.J[1] * row.J[1];
            jtj[7] += row.J[1] * row.J[2];
            jtj[8] += row.J[1] * row.J[3];
            jtj[9] += row.J[1] * row.J[4];
            jtj[10] += row.J[1] * row.J[5];

            jtj[11] += row.J[2] * row.J[2];
            jtj[12] += row.J[2] * row.J[3];
            jtj[13] += row.J[2] * row.J[4];
            jtj[14] += row.J[2] * row.J[5];

            jtj[15] += row.J[3] * row.J[3];
            jtj[16] += row.J[3] * row.J[4];
            jtj[17] += row.J[3] * row.J[5];

            jtj[18] += row.J[4] * row.J[4];
            jtj[19] += row.J[4] * row.J[5];

            jtj[20] += row.J[5] * row.J[5];

            // extra info here
            info[0] += 1;
        }
    }

    for (int i = 0; i < 32; ++i) // copy over to shared memory
        S[sline][i] = sums[i];

    __syncthreads();            // wait for everyone to finish

    if (sline < 32) { // sum up columns and copy to global memory in the final 32 threads
        for (unsigned i = 1; i < blockDim.x; ++i)
            S[0][sline] += S[i][sline];
        out[sline + blockIdx.x * 32] = S[0][sline];
    }
}

__global__ void compareVertexKernel(Image<float3> vertex1,
                                    Image<float3> vertex2,
                                    Image<float>out)
{
    const uint2 pixel = thr2pos2();

    float3 pix1=vertex1[pixel];
    float3 pix2=vertex2[pixel];

    float dist=sqrt((float) sq(pix1.x-pix2.x) +
                    (float) sq(pix1.y-pix2.y) +
                    (float) sq(pix1.z-pix2.z) );

    out[pixel]=dist;
}


//=================ICP COVARIANCE======================


__global__ void icpCovarianceFirstTerm( const Image<float3> inVertex,
                                        const Image<float3> refVertex,
                                        const Image<float3> refNormal,
                                        const Image<TrackData> trackData,
                                        Image<sMatrix6> outData,
                                        const sMatrix4 Ttrack,
                                        const sMatrix4 view,
                                        const sMatrix4 delta,
                                        const float cov_big)
{
    float Tx = delta(0,3);
    float Ty = delta(1,3);
    float  Tz = delta(2,3);

    float  roll  = atan2f(delta(2,1), delta(2,2));
    float  pitch = asinf(-delta(2,0));
    float  yaw   = atan2f(delta(1,0), delta(0,0));

    float  x, y, z, a, b, c;
    x = Tx; y = Ty; z = Tz;
    a = yaw; b = pitch; c = roll;// important // According to the rotation matrix I used and after verification, it is Yaw Pitch ROLL = [a,b,c]== [R] matrix used in the MatLab also :)

    sMatrix6 ret;

    //ICP not matched this vertex
    if(trackData.el().result!=1)
    {
        for(int i=0;i<36;i++)
            ret.data[i]=cov_big;
        outData.el()=ret;
        return;
    }


    const uint2 pixel = thr2pos2();
    const float3 projectedVertex = Ttrack * inVertex[pixel];
    const float3 projectedPos = view * projectedVertex;
    const float2 projPixel = make_float2(projectedPos.x / projectedPos.z + 0.5f,
                                         projectedPos.y / projectedPos.z + 0.5f);


    //out of bounds. add zeros
    if (projPixel.x < 0 || projPixel.x > refVertex.size.x - 1
        || projPixel.y < 0 || projPixel.y > refVertex.size.y - 1)
    {
        for(int i=0;i<36;i++)
            ret.data[i]=0.0;
        outData.el()=ret;
        return;
    }

    const uint2 refPixel = make_uint2(projPixel.x, projPixel.y);
    const float3 referenceNormal = refNormal[refPixel];

    //invalid normal
    if (referenceNormal.x == INVALID)
    {
        for(int i=0;i<36;i++)
            ret.data[i]=cov_big;
        outData.el()=ret;
        return;
    }

    float3 fp=inVertex[pixel];
    //float3 fq=invPrevPose*refVertex[refPixel];
    float3 fq=refVertex[refPixel];
    float3 fn=refNormal[refPixel];

    /*
    fp=fromVisionCordV(fp);
    fq=fromVisionCordV(fq);
    fn=fromVisionCordV(fn);
    */

    float pix=fp.x;
    float piy=fp.y;
    float piz=fp.z;

    float qix=fq.x;
    float qiy=fq.y;
    float qiz=fq.z;

    float nix=fn.x;
    float niy=fn.y;
    float niz=fn.z;

    if (niz!=niz)
    {
        outData.el()=ret;
        return;
    }
    /***********************************************************************

d2J_dX2 -- X is the [R|T] in the form of [x, y, z, a, b, c]
x, y, z is the translation part
a, b, c is the rotation part in Euler format
[x, y, z, a, b, c] is acquired from the Transformation Matrix returned by ICP.

Now d2J_dX2 is a 6x6 matrix of the form

ret(0,0)
ret(1,0)    ret(1,1)
ret(2,0)    ret(2,1)    ret(2,2)
ret(3,0)    ret(3,1)    ret(3,2)   ret(3,3)
ret(4,0)    ret(4,1)    ret(4,2)   ret(4,3)   ret(4,4)
ret(5,0)    ret(5,1)    ret(5,2)   ret(5,3)   ret(5,4)   ret(5,5)

*************************************************************************/



    /*
    d2J_dx2,     d2J_dydx,	  d2J_dzdx,   d2J_dadx,   d2J_dbdx,     d2J_dcdx,
    d2J_dxdy,    d2J_dy2,	  d2J_dzdy,   d2J_dady,   d2J_dbdy,     d2J_dcdy,
    d2J_dxdz,    d2J_dydz,    d2J_dz2,    d2J_dadz,   d2J_dbdz,     d2J_dcdz,
    d2J_dxda,    d2J_dyda,    d2J_dzda,   d2J_da2,	  d2J_dbda,     d2J_dcda,
    d2J_dxdb,    d2J_dydb,    d2J_dzdb,   d2J_dadb,   d2J_db2,      d2J_dcdb,
    d2J_dxdc,    d2J_dydc,    d2J_dzdc,   d2J_dadc,   d2J_dbdc,     d2J_dc2;
    */

    // These terms are generated from the provided Matlab scipts. We just have to copy
    // the expressions from the matlab output with two very simple changes.
    // The first one being the the sqaure of a number 'a' is shown as a^2 in matlab,
    // which is converted to pow(a,2) in the below expressions.
    // The second change is to add ';' at the end of each expression :)
    // In this way, matlab can be used to generate these terms for various objective functions of ICP
    // and they can simply be copied to the C++ files and with appropriate changes to ICP estimation,
    // its covariance can be easily estimated.

    ret(0,0) =2*pow(nix,2);


    ret(1,1) =2*pow(niy,2);


    ret(2,2) =2*pow(niz,2);


    ret(0,1) =2*nix*niy;


    ret(1,0) =2*nix*niy;


    ret(0,2) =2*nix*niz;


    ret(2,0) =2*nix*niz;


    ret(2,1) =2*niy*niz;


    ret(1,2) =2*niy*niz;


    ret(3,3) =

            (niy*(piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + pix*cos(a)*cos(b)) - nix*(piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)))*(2*niy*(piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + pix*cos(a)*cos(b)) - 2*nix*(piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a))) - (2*nix*(piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + pix*cos(a)*cos(b)) + 2*niy*(piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)))*(nix*(x - qix - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + pix*cos(a)*cos(b)) + niy*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + niz*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c)));


    ret(4,4) =

            (niy*(piz*cos(b)*cos(c)*sin(a) - pix*sin(a)*sin(b) + piy*cos(b)*sin(a)*sin(c)) - niz*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c)) + nix*(piz*cos(a)*cos(b)*cos(c) - pix*cos(a)*sin(b) + piy*cos(a)*cos(b)*sin(c)))*(2*niy*(piz*cos(b)*cos(c)*sin(a) - pix*sin(a)*sin(b) + piy*cos(b)*sin(a)*sin(c)) - 2*niz*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c)) + 2*nix*(piz*cos(a)*cos(b)*cos(c) - pix*cos(a)*sin(b) + piy*cos(a)*cos(b)*sin(c))) - (2*niy*(pix*cos(b)*sin(a) + piz*cos(c)*sin(a)*sin(b) + piy*sin(a)*sin(b)*sin(c)) + 2*niz*(piz*cos(b)*cos(c) - pix*sin(b) + piy*cos(b)*sin(c)) + 2*nix*(pix*cos(a)*cos(b) + piz*cos(a)*cos(c)*sin(b) + piy*cos(a)*sin(b)*sin(c)))*(nix*(x - qix - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + pix*cos(a)*cos(b)) + niy*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + niz*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c)));


    ret(5,5) =

            (nix*(piy*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + piz*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c))) - niy*(piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c))) + niz*(piy*cos(b)*cos(c) - piz*cos(b)*sin(c)))*(2*nix*(piy*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + piz*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c))) - 2*niy*(piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c))) + 2*niz*(piy*cos(b)*cos(c) - piz*cos(b)*sin(c))) - (2*niy*(piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b))) - 2*nix*(piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) - piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b))) + 2*niz*(piz*cos(b)*cos(c) + piy*cos(b)*sin(c)))*(nix*(x - qix - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + pix*cos(a)*cos(b)) + niy*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + niz*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c)));

    ret(3,0) =

            nix*(2*niy*(piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + pix*cos(a)*cos(b)) - 2*nix*(piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)));


    ret(0,3) =

            2*nix*(niy*(piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + pix*cos(a)*cos(b)) - nix*(piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)));


    ret(3,1) =

            niy*(2*niy*(piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + pix*cos(a)*cos(b)) - 2*nix*(piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)));


    ret(1,3) =

            2*niy*(niy*(piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + pix*cos(a)*cos(b)) - nix*(piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)));


    ret(3,2) =

            niz*(2*niy*(piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + pix*cos(a)*cos(b)) - 2*nix*(piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)));


    ret(2,3) =

            2*niz*(niy*(piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + pix*cos(a)*cos(b)) - nix*(piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)));


    ret(4,0) =

            nix*(2*niy*(piz*cos(b)*cos(c)*sin(a) - pix*sin(a)*sin(b) + piy*cos(b)*sin(a)*sin(c)) - 2*niz*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c)) + 2*nix*(piz*cos(a)*cos(b)*cos(c) - pix*cos(a)*sin(b) + piy*cos(a)*cos(b)*sin(c)));


    ret(0,4) =

            2*nix*(niy*(piz*cos(b)*cos(c)*sin(a) - pix*sin(a)*sin(b) + piy*cos(b)*sin(a)*sin(c)) - niz*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c)) + nix*(piz*cos(a)*cos(b)*cos(c) - pix*cos(a)*sin(b) + piy*cos(a)*cos(b)*sin(c)));


    ret(4,1) =

            niy*(2*niy*(piz*cos(b)*cos(c)*sin(a) - pix*sin(a)*sin(b) + piy*cos(b)*sin(a)*sin(c)) - 2*niz*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c)) + 2*nix*(piz*cos(a)*cos(b)*cos(c) - pix*cos(a)*sin(b) + piy*cos(a)*cos(b)*sin(c)));


    ret(1,4) =

            2*niy*(niy*(piz*cos(b)*cos(c)*sin(a) - pix*sin(a)*sin(b) + piy*cos(b)*sin(a)*sin(c)) - niz*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c)) + nix*(piz*cos(a)*cos(b)*cos(c) - pix*cos(a)*sin(b) + piy*cos(a)*cos(b)*sin(c)));

    ret(4,2) =

            niz*(2*niy*(piz*cos(b)*cos(c)*sin(a) - pix*sin(a)*sin(b) + piy*cos(b)*sin(a)*sin(c)) - 2*niz*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c)) + 2*nix*(piz*cos(a)*cos(b)*cos(c) - pix*cos(a)*sin(b) + piy*cos(a)*cos(b)*sin(c)));


    ret(2,4) =

            2*niz*(niy*(piz*cos(b)*cos(c)*sin(a) - pix*sin(a)*sin(b) + piy*cos(b)*sin(a)*sin(c)) - niz*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c)) + nix*(piz*cos(a)*cos(b)*cos(c) - pix*cos(a)*sin(b) + piy*cos(a)*cos(b)*sin(c)));


    ret(5,0) =

            nix*(2*nix*(piy*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + piz*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c))) - 2*niy*(piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c))) + 2*niz*(piy*cos(b)*cos(c) - piz*cos(b)*sin(c)));


    ret(0,5) =

            2*nix*(nix*(piy*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + piz*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c))) - niy*(piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c))) + niz*(piy*cos(b)*cos(c) - piz*cos(b)*sin(c)));


    ret(5,1) =

            niy*(2*nix*(piy*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + piz*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c))) - 2*niy*(piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c))) + 2*niz*(piy*cos(b)*cos(c) - piz*cos(b)*sin(c)));

    ret(1,5) =

            2*niy*(nix*(piy*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + piz*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c))) - niy*(piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c))) + niz*(piy*cos(b)*cos(c) - piz*cos(b)*sin(c)));


    ret(5,2) =

            niz*(2*nix*(piy*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + piz*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c))) - 2*niy*(piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c))) + 2*niz*(piy*cos(b)*cos(c) - piz*cos(b)*sin(c)));


    ret(2,5) =

            2*niz*(nix*(piy*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + piz*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c))) - niy*(piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c))) + niz*(piy*cos(b)*cos(c) - piz*cos(b)*sin(c)));


    ret(4,3) =

            (niy*(piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + pix*cos(a)*cos(b)) - nix*(piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)))*(2*niy*(piz*cos(b)*cos(c)*sin(a) - pix*sin(a)*sin(b) + piy*cos(b)*sin(a)*sin(c)) - 2*niz*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c)) + 2*nix*(piz*cos(a)*cos(b)*cos(c) - pix*cos(a)*sin(b) + piy*cos(a)*cos(b)*sin(c))) - (2*nix*(piz*cos(b)*cos(c)*sin(a) - pix*sin(a)*sin(b) + piy*cos(b)*sin(a)*sin(c)) - 2*niy*(piz*cos(a)*cos(b)*cos(c) - pix*cos(a)*sin(b) + piy*cos(a)*cos(b)*sin(c)))*(nix*(x - qix - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + pix*cos(a)*cos(b)) + niy*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + niz*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c)));


    ret(3,4) =

            (2*niy*(piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + pix*cos(a)*cos(b)) - 2*nix*(piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)))*(niy*(piz*cos(b)*cos(c)*sin(a) - pix*sin(a)*sin(b) + piy*cos(b)*sin(a)*sin(c)) - niz*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c)) + nix*(piz*cos(a)*cos(b)*cos(c) - pix*cos(a)*sin(b) + piy*cos(a)*cos(b)*sin(c))) - (2*nix*(piz*cos(b)*cos(c)*sin(a) - pix*sin(a)*sin(b) + piy*cos(b)*sin(a)*sin(c)) - 2*niy*(piz*cos(a)*cos(b)*cos(c) - pix*cos(a)*sin(b) + piy*cos(a)*cos(b)*sin(c)))*(nix*(x - qix - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + pix*cos(a)*cos(b)) + niy*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + niz*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c)));


    ret(5,4) =

            (2*nix*(piy*cos(a)*cos(b)*cos(c) - piz*cos(a)*cos(b)*sin(c)) - 2*niz*(piy*cos(c)*sin(b) - piz*sin(b)*sin(c)) + 2*niy*(piy*cos(b)*cos(c)*sin(a) - piz*cos(b)*sin(a)*sin(c)))*(nix*(x - qix - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + pix*cos(a)*cos(b)) + niy*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + niz*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c))) + (niy*(piz*cos(b)*cos(c)*sin(a) - pix*sin(a)*sin(b) + piy*cos(b)*sin(a)*sin(c)) - niz*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c)) + nix*(piz*cos(a)*cos(b)*cos(c) - pix*cos(a)*sin(b) + piy*cos(a)*cos(b)*sin(c)))*(2*nix*(piy*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + piz*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c))) - 2*niy*(piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c))) + 2*niz*(piy*cos(b)*cos(c) - piz*cos(b)*sin(c)));


    ret(4,5) =

            (2*nix*(piy*cos(a)*cos(b)*cos(c) - piz*cos(a)*cos(b)*sin(c)) - 2*niz*(piy*cos(c)*sin(b) - piz*sin(b)*sin(c)) + 2*niy*(piy*cos(b)*cos(c)*sin(a) - piz*cos(b)*sin(a)*sin(c)))*(nix*(x - qix - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + pix*cos(a)*cos(b)) + niy*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + niz*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c))) + (2*niy*(piz*cos(b)*cos(c)*sin(a) - pix*sin(a)*sin(b) + piy*cos(b)*sin(a)*sin(c)) - 2*niz*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c)) + 2*nix*(piz*cos(a)*cos(b)*cos(c) - pix*cos(a)*sin(b) + piy*cos(a)*cos(b)*sin(c)))*(nix*(piy*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + piz*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c))) - niy*(piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c))) + niz*(piy*cos(b)*cos(c) - piz*cos(b)*sin(c)));


    ret(3,5) =

            (2*niy*(piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + pix*cos(a)*cos(b)) - 2*nix*(piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)))*(nix*(piy*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + piz*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c))) - niy*(piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c))) + niz*(piy*cos(b)*cos(c) - piz*cos(b)*sin(c))) + (2*nix*(piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c))) + 2*niy*(piy*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + piz*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c))))*(nix*(x - qix - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + pix*cos(a)*cos(b)) + niy*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + niz*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c)));


    ret(5,3) =

            (niy*(piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + pix*cos(a)*cos(b)) - nix*(piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)))*(2*nix*(piy*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + piz*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c))) - 2*niy*(piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c))) + 2*niz*(piy*cos(b)*cos(c) - piz*cos(b)*sin(c))) + (2*nix*(piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c))) + 2*niy*(piy*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + piz*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c))))*(nix*(x - qix - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + pix*cos(a)*cos(b)) + niy*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + niz*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c)));



    outData.el()=ret;
}


__global__ void icpCovarianceSecondTerm( const Image<float3> inVertex,
                                        const Image<float3> refVertex,
                                        const Image<float3> refNormal,
                                        const Image<TrackData>  trackData,
                                        Image<sMatrix6> outData,
                                        const sMatrix4 Ttrack,
                                        const sMatrix4 view,
                                        const sMatrix4 delta,
                                        float cov_z,
                                        const float cov_big)
{

    float Tx = delta(0,3);
    float Ty = delta(1,3);
    float  Tz = delta(2,3);
    float  roll  = atan2f(delta(2,1), delta(2,2));
    float  pitch = asinf(-delta(2,0));
    float  yaw   = atan2f(delta(1,0), delta(0,0));

//    float roll  = atan2f(delta(2,1), delta(2,2));
//    float pitch = atan2f(-delta(2,0) ,sqrt( sq(delta(2,1))+sq(delta(2,2))) );
//    float yaw   = atan2f(delta(1,0), delta(0,0));


    float  x, y, z, a, b, c;
    x = Tx; y = Ty; z = Tz;
    a = yaw; b = pitch; c = roll;

    sMatrix6 d2J_dZdX_temp,mat;
    const uint2 pixel = thr2pos2();

    if(trackData[pixel].result!=1)
    {
        for(int i=0;i<36;i++)
            mat.data[i]=cov_big;
        return;
    }


    if (pixel.x >= inVertex.size.x || pixel.y >= inVertex.size.y)
    {
        for(int i=0;i<36;i++)
            mat.data[i]=cov_big;
        outData.el()=mat;
        return;
    }

    const float3 projectedVertex = Ttrack * inVertex[pixel];
    const float3 projectedPos = view * projectedVertex;
    const float2 projPixel = make_float2(projectedPos.x / projectedPos.z + 0.5f,
                                         projectedPos.y / projectedPos.z + 0.5f);


    //out of bounds. add zeros
    if (projPixel.x < 0 || projPixel.x > refVertex.size.x - 1
        || projPixel.y < 0 || projPixel.y > refVertex.size.y - 1)
    {
        for(int i=0;i<36;i++)
            mat.data[i]=0.0;
        outData.el()=mat;
        return;
    }

    const uint2 refPixel = make_uint2(projPixel.x, projPixel.y);
    const float3 referenceNormal = refNormal[refPixel];

    //invalid normal
    if (referenceNormal.x == INVALID)
    {
        for(int i=0;i<36;i++)
            mat.data[i]=cov_big;
        outData.el()=mat;
        return;
    }

    float3 fp=inVertex[pixel];
    //float3 fq=invPrevPose*refVertex[refPixel];
    float3 fq=refVertex[refPixel];
    //float3 fn=rotate(invPrevPose,modelNormals[refPixel]);
    float3 fn=refNormal[refPixel];


    /*
    fp=fromVisionCordV(fp);
    fq=fromVisionCordV(fq);
    fn=fromVisionCordV(fn);
    */

    float pix=fp.x;
    float piy=fp.y;
    float piz=fp.z;

    float qix=fq.x;
    float qiy=fq.y;
    float qiz=fq.z;

    float nix=fn.x;
    float niy=fn.y;
    float niz=fn.z;

    if (niz!=niz) // for nan removal in input point cloud data
    {
        outData.el()=mat;
        return;
    }

//         Eigen::MatrixXd d2J_dZdX_temp(6,6);


        /*
        float 	d2J_dZdX_temp(0,0),    d2J_dZdX_temp(0,1),	d2J_dZdX_temp(0,2),  	   d2J_dZdX_temp(0,3),    d2J_dZdX_temp(0,4),	   d2J_dZdX_temp(0,5),
                d2J_dZdX_temp(1,0),    d2J_dZdX_temp(1,1),	d2J_dZdX_temp(1,2),   	   d2J_dZdX_temp(1,3),    d2J_dZdX_temp(1,4),	   d2J_dZdX_temp(1,5),
                d2J_dZdX_temp(2,0),    d2J_dZdX_temp(2,1),    d2J_dZdX_temp(2,2),       d2J_dZdX_temp(2,3),    d2J_dZdX_temp(2,4),    d2J_dZdX_temp(2,5),
                d2J_dZdX_temp(3,0),    d2J_dZdX_temp(3,1),    d2J_dZdX_temp(3,2),       d2J_dZdX_temp(3,3),    d2J_dZdX_temp(3,4),    d2J_dZdX_temp(3,5),
                d2J_dZdX_temp(4,0),    d2J_dZdX_temp(4,1),    d2J_dZdX_temp(4,2),       d2J_dZdX_temp(4,3),    d2J_dZdX_temp(4,4),    d2J_dZdX_temp(4,5),
                d2J_dZdX_temp(5,0),    d2J_dZdX_temp(5,1),    d2J_dZdX_temp(5,2),       d2J_dZdX_temp(5,3),    d2J_dZdX_temp(5,4),    d2J_dZdX_temp(5,5);
        */

    d2J_dZdX_temp(0,0) =

            2*nix*(nix*cos(a)*cos(b) - niz*sin(b) + niy*cos(b)*sin(a));


    d2J_dZdX_temp(1,0) =

            2*niy*(nix*cos(a)*cos(b) - niz*sin(b) + niy*cos(b)*sin(a));


    d2J_dZdX_temp(2,0) =

            2*niz*(nix*cos(a)*cos(b) - niz*sin(b) + niy*cos(b)*sin(a));


    d2J_dZdX_temp(3,0) =

            (2*niy*cos(a)*cos(b) - 2*nix*cos(b)*sin(a))*(nix*(x - qix - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + pix*cos(a)*cos(b)) + niy*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + niz*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c))) + (2*niy*(piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + pix*cos(a)*cos(b)) - 2*nix*(piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)))*(nix*cos(a)*cos(b) - niz*sin(b) + niy*cos(b)*sin(a));


    d2J_dZdX_temp(4,0) =

            (2*niy*(piz*cos(b)*cos(c)*sin(a) - pix*sin(a)*sin(b) + piy*cos(b)*sin(a)*sin(c)) - 2*niz*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c)) + 2*nix*(piz*cos(a)*cos(b)*cos(c) - pix*cos(a)*sin(b) + piy*cos(a)*cos(b)*sin(c)))*(nix*cos(a)*cos(b) - niz*sin(b) + niy*cos(b)*sin(a)) - (2*niz*cos(b) + 2*nix*cos(a)*sin(b) + 2*niy*sin(a)*sin(b))*(nix*(x - qix - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + pix*cos(a)*cos(b)) + niy*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + niz*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c)));


    d2J_dZdX_temp(5,0) =

            (2*nix*(piy*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + piz*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c))) - 2*niy*(piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c))) + 2*niz*(piy*cos(b)*cos(c) - piz*cos(b)*sin(c)))*(nix*cos(a)*cos(b) - niz*sin(b) + niy*cos(b)*sin(a));


    d2J_dZdX_temp(0,1) =

            2*nix*(niy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - nix*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + niz*cos(b)*sin(c));


    d2J_dZdX_temp(1,1) =

            2*niy*(niy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - nix*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + niz*cos(b)*sin(c));


    d2J_dZdX_temp(2,1) =

            2*niz*(niy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - nix*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + niz*cos(b)*sin(c));


    d2J_dZdX_temp(3,1) =

            (2*niy*(piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + pix*cos(a)*cos(b)) - 2*nix*(piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)))*(niy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - nix*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + niz*cos(b)*sin(c)) - (2*nix*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) + 2*niy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)))*(nix*(x - qix - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + pix*cos(a)*cos(b)) + niy*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + niz*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c)));


    d2J_dZdX_temp(4,1) =

            (2*niy*(piz*cos(b)*cos(c)*sin(a) - pix*sin(a)*sin(b) + piy*cos(b)*sin(a)*sin(c)) - 2*niz*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c)) + 2*nix*(piz*cos(a)*cos(b)*cos(c) - pix*cos(a)*sin(b) + piy*cos(a)*cos(b)*sin(c)))*(niy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - nix*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + niz*cos(b)*sin(c)) + (2*nix*cos(a)*cos(b)*sin(c) - 2*niz*sin(b)*sin(c) + 2*niy*cos(b)*sin(a)*sin(c))*(nix*(x - qix - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + pix*cos(a)*cos(b)) + niy*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + niz*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c)));


    d2J_dZdX_temp(5,1) =

            (2*nix*(piy*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + piz*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c))) - 2*niy*(piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c))) + 2*niz*(piy*cos(b)*cos(c) - piz*cos(b)*sin(c)))*(niy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - nix*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + niz*cos(b)*sin(c)) + (2*nix*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - 2*niy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + 2*niz*cos(b)*cos(c))*(nix*(x - qix - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + pix*cos(a)*cos(b)) + niy*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + niz*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c)));


    d2J_dZdX_temp(0,2) =

            2*nix*(nix*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - niy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + niz*cos(b)*cos(c));


    d2J_dZdX_temp(1,2) =

            2*niy*(nix*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - niy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + niz*cos(b)*cos(c));


    d2J_dZdX_temp(2,2) =

            2*niz*(nix*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - niy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + niz*cos(b)*cos(c));


    d2J_dZdX_temp(3,2) =

            (2*niy*(piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + pix*cos(a)*cos(b)) - 2*nix*(piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)))*(nix*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - niy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + niz*cos(b)*cos(c)) + (2*nix*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + 2*niy*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)))*(nix*(x - qix - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + pix*cos(a)*cos(b)) + niy*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + niz*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c)));


    d2J_dZdX_temp(4,2) =

            (2*niy*(piz*cos(b)*cos(c)*sin(a) - pix*sin(a)*sin(b) + piy*cos(b)*sin(a)*sin(c)) - 2*niz*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c)) + 2*nix*(piz*cos(a)*cos(b)*cos(c) - pix*cos(a)*sin(b) + piy*cos(a)*cos(b)*sin(c)))*(nix*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - niy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + niz*cos(b)*cos(c)) + (2*nix*cos(a)*cos(b)*cos(c) - 2*niz*cos(c)*sin(b) + 2*niy*cos(b)*cos(c)*sin(a))*(nix*(x - qix - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + pix*cos(a)*cos(b)) + niy*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + niz*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c)));


    d2J_dZdX_temp(5,2) =

            (2*nix*(piy*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + piz*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c))) - 2*niy*(piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c))) + 2*niz*(piy*cos(b)*cos(c) - piz*cos(b)*sin(c)))*(nix*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - niy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + niz*cos(b)*cos(c)) - (2*niy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - 2*nix*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + 2*niz*cos(b)*sin(c))*(nix*(x - qix - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + pix*cos(a)*cos(b)) + niy*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + niz*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c)));


    d2J_dZdX_temp(0,3) =

            -2*pow(nix,2);


    d2J_dZdX_temp(1,3) =

            -2*nix*niy;


    d2J_dZdX_temp(2,3) =

            -2*nix*niz;


    d2J_dZdX_temp(3,3) =

            -nix*(2*niy*(piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + pix*cos(a)*cos(b)) - 2*nix*(piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)));


    d2J_dZdX_temp(4,3) =

            -nix*(2*niy*(piz*cos(b)*cos(c)*sin(a) - pix*sin(a)*sin(b) + piy*cos(b)*sin(a)*sin(c)) - 2*niz*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c)) + 2*nix*(piz*cos(a)*cos(b)*cos(c) - pix*cos(a)*sin(b) + piy*cos(a)*cos(b)*sin(c)));


    d2J_dZdX_temp(5,3) =

            -nix*(2*nix*(piy*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + piz*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c))) - 2*niy*(piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c))) + 2*niz*(piy*cos(b)*cos(c) - piz*cos(b)*sin(c)));


    d2J_dZdX_temp(0,4) =

            -2*nix*niy;


    d2J_dZdX_temp(1,4) =

            -2*pow(niy,2);


    d2J_dZdX_temp(2,4) =

            -2*niy*niz;


    d2J_dZdX_temp(3,4) =

            -niy*(2*niy*(piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + pix*cos(a)*cos(b)) - 2*nix*(piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)));


    d2J_dZdX_temp(4,4) =

            -niy*(2*niy*(piz*cos(b)*cos(c)*sin(a) - pix*sin(a)*sin(b) + piy*cos(b)*sin(a)*sin(c)) - 2*niz*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c)) + 2*nix*(piz*cos(a)*cos(b)*cos(c) - pix*cos(a)*sin(b) + piy*cos(a)*cos(b)*sin(c)));


    d2J_dZdX_temp(5,4) =

            -niy*(2*nix*(piy*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + piz*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c))) - 2*niy*(piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c))) + 2*niz*(piy*cos(b)*cos(c) - piz*cos(b)*sin(c)));


    d2J_dZdX_temp(0,5) =

            -2*nix*niz;


    d2J_dZdX_temp(1,5) =

            -2*niy*niz;


    d2J_dZdX_temp(2,5) =

            -2*pow(niz,2);


    d2J_dZdX_temp(3,5) =

            -niz*(2*niy*(piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + pix*cos(a)*cos(b)) - 2*nix*(piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)));


    d2J_dZdX_temp(4,5) =

            -niz*(2*niy*(piz*cos(b)*cos(c)*sin(a) - pix*sin(a)*sin(b) + piy*cos(b)*sin(a)*sin(c)) - 2*niz*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c)) + 2*nix*(piz*cos(a)*cos(b)*cos(c) - pix*cos(a)*sin(b) + piy*cos(a)*cos(b)*sin(c)));


    d2J_dZdX_temp(5,5) =

            -niz*(2*nix*(piy*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + piz*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c))) - 2*niy*(piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c))) + 2*niz*(piy*cos(b)*cos(c) - piz*cos(b)*sin(c)));

    for(int i=0;i<36;i++)
        mat.data[i]=0.0;

    for(int i=0;i<6;i++)
    {
        for(int j=0;j<6;j++)
        {
            for(int k=0;k<6;k++)
            {
                mat(i,j) += d2J_dZdX_temp(i,k) * d2J_dZdX_temp(j,k) * cov_z;
            }
        }
    }
    outData.el()=mat;
}


//===========================================Point to point covariance ===========================
__global__ void point2PointCovFirstTerm(const float3 *vert,
                                        int vertSize,
                                        const float3 *prevVert,
                                        int prevVertSize,
                                        const int *sourceCorr,
                                        const int *targetCorr,
                                        int correspSize,
                                        sMatrix4 delta,
                                        sMatrix6 *outData,
                                        const float cov_big)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx>=correspSize)
        return;

    float Tx = delta(0,3);
    float Ty = delta(1,3);
    float  Tz = delta(2,3);

    float  roll  = atan2f(delta(2,1), delta(2,2));
    float  pitch = asinf(-delta(2,0));
    float  yaw   = atan2f(delta(1,0), delta(0,0));


    float  x, y, z, a, b, c;
    x = Tx; y = Ty; z = Tz;
    a = yaw; b = pitch; c = roll;// important // According to the rotation matrix I used and after verification, it is Yaw Pitch ROLL = [a,b,c]== [R] matrix used in the MatLab also :)

    sMatrix6 ret;
    /*
    int2 pair=corresp[idx];

    if(pair.x<0 || pair.x>=vertSize)
    {
        for(int i=0;i<36;i++)
            ret.data[i]=cov_big;
        outData[idx]=ret;
        return;
    }

    if(pair.y<0 || pair.y>=prevVertSize )
    {
        for(int i=0;i<36;i++)
            ret.data[i]=cov_big;
        outData[idx]=ret;
        return;
    }
    */
    int sourceIdx=sourceCorr[idx];
    int targetIdx=targetCorr[idx];

    if(sourceIdx<0 || sourceIdx>=vertSize)
    {
        for(int i=0;i<36;i++)
            ret.data[i]=cov_big;
        outData[idx]=ret;
        return;
    }

    if(targetIdx<0 || targetIdx>=prevVertSize )
    {
        for(int i=0;i<36;i++)
            ret.data[i]=cov_big;
//         printf("ASDFGSDFHSFHSDFH2\n");
        outData[idx]=ret;
        return;
    }

    /*
    float3 fp=vert[sourceIdx];
    float3 fq=prevVert[targetIdx];
    */

    float3 fq=vert[sourceIdx];
    float3 fp=prevVert[targetIdx];

    float pix=fp.x;
    float piy=fp.y;
    float piz=fp.z;

    float qix=fq.x;
    float qiy=fq.y;
    float qiz=fq.z;



    /*
    d2J_dx2,     d2J_dydx,	  d2J_dzdx,   d2J_dadx,   d2J_dbdx,     d2J_dcdx,
    d2J_dxdy,    d2J_dy2,	  d2J_dzdy,   d2J_dady,   d2J_dbdy,     d2J_dcdy,
    d2J_dxdz,    d2J_dydz,    d2J_dz2,    d2J_dadz,   d2J_dbdz,     d2J_dcdz,
    d2J_dxda,    d2J_dyda,    d2J_dzda,   d2J_da2,	  d2J_dbda,     d2J_dcda,
    d2J_dxdb,    d2J_dydb,    d2J_dzdb,   d2J_dadb,   d2J_db2,      d2J_dcdb,
    d2J_dxdc,    d2J_dydc,    d2J_dzdc,   d2J_dadc,   d2J_dbdc,     d2J_dc2;
    */
    float d2J_dx2 =

    2;


    float d2J_dy2 =

    2;


    float d2J_dz2 =

    2;


    float d2J_dydx =

    0;


    float d2J_dxdy =

    0;


    float d2J_dzdx =

    0;


    float d2J_dxdz =

    0;


    float d2J_dydz =

    0;


    float d2J_dzdy =

    0;


    float d2J_da2 =

    (piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + pix*cos(a)*cos(b))*(2*piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - 2*piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + 2*pix*cos(a)*cos(b)) - (2*piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - 2*piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + 2*pix*cos(b)*sin(a))*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + (piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a))*(2*piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - 2*piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + 2*pix*cos(b)*sin(a)) - (2*piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - 2*piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + 2*pix*cos(a)*cos(b))*(x - qix - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + pix*cos(a)*cos(b));


    float d2J_db2 =

    (pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c))*(2*pix*cos(b) + 2*piz*cos(c)*sin(b) + 2*piy*sin(b)*sin(c)) - (2*piz*cos(b)*cos(c) - 2*pix*sin(b) + 2*piy*cos(b)*sin(c))*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c)) - (2*pix*cos(a)*cos(b) + 2*piz*cos(a)*cos(c)*sin(b) + 2*piy*cos(a)*sin(b)*sin(c))*(x - qix - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + pix*cos(a)*cos(b)) + (piz*cos(a)*cos(b)*cos(c) - pix*cos(a)*sin(b) + piy*cos(a)*cos(b)*sin(c))*(2*piz*cos(a)*cos(b)*cos(c) - 2*pix*cos(a)*sin(b) + 2*piy*cos(a)*cos(b)*sin(c)) - (2*pix*cos(b)*sin(a) + 2*piz*cos(c)*sin(a)*sin(b) + 2*piy*sin(a)*sin(b)*sin(c))*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + (piz*cos(b)*cos(c)*sin(a) - pix*sin(a)*sin(b) + piy*cos(b)*sin(a)*sin(c))*(2*piz*cos(b)*cos(c)*sin(a) - 2*pix*sin(a)*sin(b) + 2*piy*cos(b)*sin(a)*sin(c));


    float d2J_dc2 =

    (piy*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + piz*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)))*(2*piy*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + 2*piz*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c))) + (piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)))*(2*piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + 2*piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c))) - (2*piz*cos(b)*cos(c) + 2*piy*cos(b)*sin(c))*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c)) + (2*piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) - 2*piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)))*(x - qix - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + pix*cos(a)*cos(b)) + (piy*cos(b)*cos(c) - piz*cos(b)*sin(c))*(2*piy*cos(b)*cos(c) - 2*piz*cos(b)*sin(c)) - (2*piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - 2*piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)))*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a));


    float d2J_dxda =

    2*piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) - 2*piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - 2*pix*cos(b)*sin(a);


    float d2J_dadx =

    2*piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) - 2*piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - 2*pix*cos(b)*sin(a);


    float d2J_dyda =

    2*piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - 2*piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + 2*pix*cos(a)*cos(b);


    float d2J_dady =

    2*piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - 2*piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + 2*pix*cos(a)*cos(b);


    float d2J_dzda =

    0;


    float d2J_dadz =

    0;


    float d2J_dxdb =

    2*piz*cos(a)*cos(b)*cos(c) - 2*pix*cos(a)*sin(b) + 2*piy*cos(a)*cos(b)*sin(c);


    float d2J_dbdx =

    2*piz*cos(a)*cos(b)*cos(c) - 2*pix*cos(a)*sin(b) + 2*piy*cos(a)*cos(b)*sin(c);


    float d2J_dydb =

    2*piz*cos(b)*cos(c)*sin(a) - 2*pix*sin(a)*sin(b) + 2*piy*cos(b)*sin(a)*sin(c);


    float d2J_dbdy =

    2*piz*cos(b)*cos(c)*sin(a) - 2*pix*sin(a)*sin(b) + 2*piy*cos(b)*sin(a)*sin(c);


    float d2J_dzdb =

    - 2*pix*cos(b) - 2*piz*cos(c)*sin(b) - 2*piy*sin(b)*sin(c);


    float d2J_dbdz =

    - 2*pix*cos(b) - 2*piz*cos(c)*sin(b) - 2*piy*sin(b)*sin(c);


    float d2J_dxdc =

    2*piy*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + 2*piz*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c));


    float d2J_dcdx =

    2*piy*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + 2*piz*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c));


    float d2J_dydc =

    - 2*piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) - 2*piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c));


    float d2J_dcdy =

    - 2*piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) - 2*piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c));


    float d2J_dzdc =

    2*piy*cos(b)*cos(c) - 2*piz*cos(b)*sin(c);


    float d2J_dcdz =

    2*piy*cos(b)*cos(c) - 2*piz*cos(b)*sin(c);


    float d2J_dadb =

    (2*piz*cos(b)*cos(c)*sin(a) - 2*pix*sin(a)*sin(b) + 2*piy*cos(b)*sin(a)*sin(c))*(piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + pix*cos(a)*cos(b)) - (2*piz*cos(a)*cos(b)*cos(c) - 2*pix*cos(a)*sin(b) + 2*piy*cos(a)*cos(b)*sin(c))*(piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + (2*piz*cos(a)*cos(b)*cos(c) - 2*pix*cos(a)*sin(b) + 2*piy*cos(a)*cos(b)*sin(c))*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) - (2*piz*cos(b)*cos(c)*sin(a) - 2*pix*sin(a)*sin(b) + 2*piy*cos(b)*sin(a)*sin(c))*(x - qix - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + pix*cos(a)*cos(b));


    float d2J_dbda =

    (piz*cos(b)*cos(c)*sin(a) - pix*sin(a)*sin(b) + piy*cos(b)*sin(a)*sin(c))*(2*piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - 2*piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + 2*pix*cos(a)*cos(b)) - (piz*cos(a)*cos(b)*cos(c) - pix*cos(a)*sin(b) + piy*cos(a)*cos(b)*sin(c))*(2*piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - 2*piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + 2*pix*cos(b)*sin(a)) + (2*piz*cos(a)*cos(b)*cos(c) - 2*pix*cos(a)*sin(b) + 2*piy*cos(a)*cos(b)*sin(c))*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) - (2*piz*cos(b)*cos(c)*sin(a) - 2*pix*sin(a)*sin(b) + 2*piy*cos(b)*sin(a)*sin(c))*(x - qix - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + pix*cos(a)*cos(b));


    float d2J_dbdc =

    (2*piy*cos(a)*cos(b)*cos(c) - 2*piz*cos(a)*cos(b)*sin(c))*(x - qix - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + pix*cos(a)*cos(b)) + (2*piy*cos(b)*cos(c)*sin(a) - 2*piz*cos(b)*sin(a)*sin(c))*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) - (2*piy*cos(b)*cos(c) - 2*piz*cos(b)*sin(c))*(pix*cos(b) + piz*cos(c)*sin(b) + piy*sin(b)*sin(c)) - (2*piy*cos(c)*sin(b) - 2*piz*sin(b)*sin(c))*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c)) + (2*piy*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + 2*piz*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)))*(piz*cos(a)*cos(b)*cos(c) - pix*cos(a)*sin(b) + piy*cos(a)*cos(b)*sin(c)) - (2*piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + 2*piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)))*(piz*cos(b)*cos(c)*sin(a) - pix*sin(a)*sin(b) + piy*cos(b)*sin(a)*sin(c));


    float d2J_dcdb =

    (2*piy*cos(a)*cos(b)*cos(c) - 2*piz*cos(a)*cos(b)*sin(c))*(x - qix - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + pix*cos(a)*cos(b)) + (2*piy*cos(b)*cos(c)*sin(a) - 2*piz*cos(b)*sin(a)*sin(c))*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) - (piy*cos(b)*cos(c) - piz*cos(b)*sin(c))*(2*pix*cos(b) + 2*piz*cos(c)*sin(b) + 2*piy*sin(b)*sin(c)) - (2*piy*cos(c)*sin(b) - 2*piz*sin(b)*sin(c))*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c)) + (piy*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + piz*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)))*(2*piz*cos(a)*cos(b)*cos(c) - 2*pix*cos(a)*sin(b) + 2*piy*cos(a)*cos(b)*sin(c)) - (piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)))*(2*piz*cos(b)*cos(c)*sin(a) - 2*pix*sin(a)*sin(b) + 2*piy*cos(b)*sin(a)*sin(c));


    float d2J_dcda =

    (2*piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + 2*piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)))*(x - qix - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + pix*cos(a)*cos(b)) - (piy*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + piz*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)))*(2*piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - 2*piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + 2*pix*cos(b)*sin(a)) - (piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)))*(2*piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - 2*piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + 2*pix*cos(a)*cos(b)) + (2*piy*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + 2*piz*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)))*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a));


    float d2J_dadc =

    (2*piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + 2*piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)))*(x - qix - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + pix*cos(a)*cos(b)) - (2*piy*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + 2*piz*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)))*(piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) - (2*piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + 2*piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)))*(piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + pix*cos(a)*cos(b)) + (2*piy*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + 2*piz*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)))*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a));


    ret=sMatrix6( d2J_dx2,     d2J_dydx,  d2J_dzdx,   d2J_dadx,   d2J_dbdx,     d2J_dcdx,
                  d2J_dxdy,    d2J_dy2,	  d2J_dzdy,   d2J_dady,   d2J_dbdy,     d2J_dcdy,
                  d2J_dxdz,    d2J_dydz,  d2J_dz2,    d2J_dadz,   d2J_dbdz,     d2J_dcdz,
                  d2J_dxda,    d2J_dyda,  d2J_dzda,   d2J_da2,	  d2J_dbda,     d2J_dcda,
                  d2J_dxdb,    d2J_dydb,  d2J_dzdb,   d2J_dadb,   d2J_db2,      d2J_dcdb,
                  d2J_dxdc,    d2J_dydc,  d2J_dzdc,   d2J_dadc,   d2J_dbdc,     d2J_dc2 );



    outData[idx]=ret;
}

__global__ void point2PointCovSecondTerm(const float3 *vert,
                                        int vertSize,
                                        const float3 *prevVert,
                                        int prevVertSize,
                                        const int *sourceCorr,
                                        const int *targetCorr,
                                        int correspSize,
                                        const sMatrix4 delta,
                                        float cov_z,
                                        sMatrix6 *outData,
                                        const float cov_big)
{

    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx>=correspSize)
        return;

    float Tx = delta(0,3);
    float Ty = delta(1,3);
    float  Tz = delta(2,3);

    float  roll  = atan2f(delta(2,1), delta(2,2));
    float  pitch = asinf(-delta(2,0));
    float  yaw   = atan2f(delta(1,0), delta(0,0));


    float  x, y, z, a, b, c;
    x = Tx; y = Ty; z = Tz;
    a = yaw; b = pitch; c = roll;// important // According to the rotation matrix I used and after verification, it is Yaw Pitch ROLL = [a,b,c]== [R] matrix used in the MatLab also :)

    sMatrix6 ret;
    /*
    int2 pair=corresp[idx];

    if(pair.x<0 || pair.x>=vertSize)
    {
        for(int i=0;i<36;i++)
            ret.data[i]=cov_big;
        outData[idx]=ret;
        return;
    }

    if(pair.y<0 || pair.y>=prevVertSize )
    {
        for(int i=0;i<36;i++)
            ret.data[i]=cov_big;
        outData[idx]=ret;
        return;
    }

    float3 fp=vert[pair.x];
    float3 fq=prevVert[pair.y];
    */

    int sourceIdx=sourceCorr[idx];
    int targetIdx=targetCorr[idx];

    if(sourceIdx<0 || sourceIdx>=vertSize)
    {
        for(int i=0;i<36;i++)
            ret.data[i]=cov_big;
        outData[idx]=ret;
        return;
    }

    if(targetIdx<0 || targetIdx>=prevVertSize )
    {
        for(int i=0;i<36;i++)
            ret.data[i]=cov_big;
        outData[idx]=ret;
        return;
    }

    /*
    float3 fp=vert[sourceIdx];
    float3 fq=prevVert[targetIdx];
    */


    float3 fq=vert[sourceIdx];
    float3 fp=prevVert[targetIdx];

    float pix=fp.x;
    float piy=fp.y;
    float piz=fp.z;

    float qix=fq.x;
    float qiy=fq.y;
    float qiz=fq.z;

    /*
    d2J_dpix_dx,    d2J_dpiy_dx,	d2J_dpiz_dx,  	   d2J_dqix_dx,    d2J_dqiy_dx,	   d2J_dqiz_dx,
    d2J_dpix_dy,    d2J_dpiy_dy,	d2J_dpiz_dy,   	   d2J_dqix_dy,    d2J_dqiy_dy,	   d2J_dqiz_dy,
    d2J_dpix_dz,    d2J_dpiy_dz,    d2J_dpiz_dz,       d2J_dqix_dz,    d2J_dqiy_dz,    d2J_dqiz_dz,
    d2J_dpix_da,    d2J_dpiy_da,    d2J_dpiz_da,       d2J_dqix_da,    d2J_dqiy_da,    d2J_dqiz_da,
    d2J_dpix_db,    d2J_dpiy_db,    d2J_dpiz_db,       d2J_dqix_db,    d2J_dqiy_db,    d2J_dqiz_db,
    d2J_dpix_dc,    d2J_dpiy_dc,    d2J_dpiz_dc,       d2J_dqix_dc,    d2J_dqiy_dc,    d2J_dqiz_dc;
    */

    sMatrix6 tmp;
    float d2J_dpix_dx =

    2*cos(a)*cos(b);


    float d2J_dpix_dy =

    2*cos(b)*sin(a);


    float d2J_dpix_dz =

    -2*sin(b);


    float d2J_dpix_da =

    cos(b)*sin(a)*(2*piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - 2*piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + 2*pix*cos(a)*cos(b)) - cos(a)*cos(b)*(2*piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - 2*piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + 2*pix*cos(b)*sin(a)) - 2*cos(b)*sin(a)*(x - qix - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + pix*cos(a)*cos(b)) + 2*cos(a)*cos(b)*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a));


    float d2J_dpix_db =

    sin(b)*(2*pix*cos(b) + 2*piz*cos(c)*sin(b) + 2*piy*sin(b)*sin(c)) - 2*cos(b)*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c)) + cos(a)*cos(b)*(2*piz*cos(a)*cos(b)*cos(c) - 2*pix*cos(a)*sin(b) + 2*piy*cos(a)*cos(b)*sin(c)) - 2*sin(a)*sin(b)*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + cos(b)*sin(a)*(2*piz*cos(b)*cos(c)*sin(a) - 2*pix*sin(a)*sin(b) + 2*piy*cos(b)*sin(a)*sin(c)) - 2*cos(a)*sin(b)*(x - qix - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + pix*cos(a)*cos(b));


    float d2J_dpix_dc =

    cos(a)*cos(b)*(2*piy*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + 2*piz*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c))) - sin(b)*(2*piy*cos(b)*cos(c) - 2*piz*cos(b)*sin(c)) - cos(b)*sin(a)*(2*piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + 2*piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)));


    float d2J_dpiy_dx =

    2*cos(a)*sin(b)*sin(c) - 2*cos(c)*sin(a);


    float d2J_dpiy_dy =

    2*cos(a)*cos(c) + 2*sin(a)*sin(b)*sin(c);


    float d2J_dpiy_dz =

    2*cos(b)*sin(c);


    float d2J_dpiy_da =

    (cos(a)*cos(c) + sin(a)*sin(b)*sin(c))*(2*piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - 2*piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + 2*pix*cos(a)*cos(b)) + (cos(c)*sin(a) - cos(a)*sin(b)*sin(c))*(2*piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - 2*piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + 2*pix*cos(b)*sin(a)) - (2*cos(a)*cos(c) + 2*sin(a)*sin(b)*sin(c))*(x - qix - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + pix*cos(a)*cos(b)) - (2*cos(c)*sin(a) - 2*cos(a)*sin(b)*sin(c))*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a));


    float d2J_dpiy_db =

    (cos(a)*cos(c) + sin(a)*sin(b)*sin(c))*(2*piz*cos(b)*cos(c)*sin(a) - 2*pix*sin(a)*sin(b) + 2*piy*cos(b)*sin(a)*sin(c)) - (cos(c)*sin(a) - cos(a)*sin(b)*sin(c))*(2*piz*cos(a)*cos(b)*cos(c) - 2*pix*cos(a)*sin(b) + 2*piy*cos(a)*cos(b)*sin(c)) - 2*sin(b)*sin(c)*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c)) - cos(b)*sin(c)*(2*pix*cos(b) + 2*piz*cos(c)*sin(b) + 2*piy*sin(b)*sin(c)) + 2*cos(a)*cos(b)*sin(c)*(x - qix - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + pix*cos(a)*cos(b)) + 2*cos(b)*sin(a)*sin(c)*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a));


    float d2J_dpiy_dc =

    (2*sin(a)*sin(c) + 2*cos(a)*cos(c)*sin(b))*(x - qix - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + pix*cos(a)*cos(b)) - (2*cos(a)*sin(c) - 2*cos(c)*sin(a)*sin(b))*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) - (cos(a)*cos(c) + sin(a)*sin(b)*sin(c))*(2*piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + 2*piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c))) - (cos(c)*sin(a) - cos(a)*sin(b)*sin(c))*(2*piy*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + 2*piz*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c))) + 2*cos(b)*cos(c)*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c)) + cos(b)*sin(c)*(2*piy*cos(b)*cos(c) - 2*piz*cos(b)*sin(c));


    float d2J_dpiz_dx =

    2*sin(a)*sin(c) + 2*cos(a)*cos(c)*sin(b);


    float d2J_dpiz_dy =

    2*cos(c)*sin(a)*sin(b) - 2*cos(a)*sin(c);


    float d2J_dpiz_dz =

    2*cos(b)*cos(c);


    float d2J_dpiz_da =

    (2*cos(a)*sin(c) - 2*cos(c)*sin(a)*sin(b))*(x - qix - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + pix*cos(a)*cos(b)) - (sin(a)*sin(c) + cos(a)*cos(c)*sin(b))*(2*piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - 2*piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + 2*pix*cos(b)*sin(a)) - (cos(a)*sin(c) - cos(c)*sin(a)*sin(b))*(2*piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - 2*piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + 2*pix*cos(a)*cos(b)) + (2*sin(a)*sin(c) + 2*cos(a)*cos(c)*sin(b))*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a));


    float d2J_dpiz_db =

    (sin(a)*sin(c) + cos(a)*cos(c)*sin(b))*(2*piz*cos(a)*cos(b)*cos(c) - 2*pix*cos(a)*sin(b) + 2*piy*cos(a)*cos(b)*sin(c)) - (cos(a)*sin(c) - cos(c)*sin(a)*sin(b))*(2*piz*cos(b)*cos(c)*sin(a) - 2*pix*sin(a)*sin(b) + 2*piy*cos(b)*sin(a)*sin(c)) - 2*cos(c)*sin(b)*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c)) - cos(b)*cos(c)*(2*pix*cos(b) + 2*piz*cos(c)*sin(b) + 2*piy*sin(b)*sin(c)) + 2*cos(a)*cos(b)*cos(c)*(x - qix - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + pix*cos(a)*cos(b)) + 2*cos(b)*cos(c)*sin(a)*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a));


    float d2J_dpiz_dc =

    (2*cos(c)*sin(a) - 2*cos(a)*sin(b)*sin(c))*(x - qix - piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) + piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + pix*cos(a)*cos(b)) - (2*cos(a)*cos(c) + 2*sin(a)*sin(b)*sin(c))*(y - qiy + piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + pix*cos(b)*sin(a)) + (sin(a)*sin(c) + cos(a)*cos(c)*sin(b))*(2*piy*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) + 2*piz*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c))) + (cos(a)*sin(c) - cos(c)*sin(a)*sin(b))*(2*piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + 2*piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c))) + cos(b)*cos(c)*(2*piy*cos(b)*cos(c) - 2*piz*cos(b)*sin(c)) - 2*cos(b)*sin(c)*(z - qiz - pix*sin(b) + piz*cos(b)*cos(c) + piy*cos(b)*sin(c));


    float d2J_dqix_dx =

    -2;


    float d2J_dqix_dy =

    0;


    float d2J_dqix_dz =

    0;


    float d2J_dqix_da =

    2*piy*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c)) - 2*piz*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + 2*pix*cos(b)*sin(a);


    float d2J_dqix_db =

    2*pix*cos(a)*sin(b) - 2*piz*cos(a)*cos(b)*cos(c) - 2*piy*cos(a)*cos(b)*sin(c);


    float d2J_dqix_dc =

    - 2*piy*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - 2*piz*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c));


    float d2J_dqiy_dx =

    0;


    float d2J_dqiy_dy =

    -2;


    float d2J_dqiy_dz =

    0;


    float d2J_dqiy_da =

    2*piy*(cos(c)*sin(a) - cos(a)*sin(b)*sin(c)) - 2*piz*(sin(a)*sin(c) + cos(a)*cos(c)*sin(b)) - 2*pix*cos(a)*cos(b);


    float d2J_dqiy_db =

    2*pix*sin(a)*sin(b) - 2*piz*cos(b)*cos(c)*sin(a) - 2*piy*cos(b)*sin(a)*sin(c);


    float d2J_dqiy_dc =

    2*piy*(cos(a)*sin(c) - cos(c)*sin(a)*sin(b)) + 2*piz*(cos(a)*cos(c) + sin(a)*sin(b)*sin(c));


    float d2J_dqiz_dx =

    0;


    float d2J_dqiz_dy =

    0;


    float d2J_dqiz_dz =

    -2;


    float d2J_dqiz_da =

    0;


    float d2J_dqiz_db =

    2*pix*cos(b) + 2*piz*cos(c)*sin(b) + 2*piy*sin(b)*sin(c);


    float d2J_dqiz_dc =

    2*piz*cos(b)*sin(c) - 2*piy*cos(b)*cos(c);

    tmp=sMatrix6(
            d2J_dpix_dx,    d2J_dpiy_dx,	d2J_dpiz_dx,  	   d2J_dqix_dx,    d2J_dqiy_dx,	   d2J_dqiz_dx,
            d2J_dpix_dy,    d2J_dpiy_dy,	d2J_dpiz_dy,   	   d2J_dqix_dy,    d2J_dqiy_dy,	   d2J_dqiz_dy,
            d2J_dpix_dz,    d2J_dpiy_dz,    d2J_dpiz_dz,       d2J_dqix_dz,    d2J_dqiy_dz,    d2J_dqiz_dz,
            d2J_dpix_da,    d2J_dpiy_da,    d2J_dpiz_da,       d2J_dqix_da,    d2J_dqiy_da,    d2J_dqiz_da,
            d2J_dpix_db,    d2J_dpiy_db,    d2J_dpiz_db,       d2J_dqix_db,    d2J_dqiy_db,    d2J_dqiz_db,
            d2J_dpix_dc,    d2J_dpiy_dc,    d2J_dpiz_dc,       d2J_dqix_dc,    d2J_dqiy_dc,    d2J_dqiz_dc);

    sMatrix6 mat;
    for(int i=0;i<36;i++)
        mat.data[i]=0.0;

    for(int i=0;i<6;i++)
    {
        for(int j=0;j<6;j++)
        {
            for(int k=0;k<6;k++)
            {
                mat(i,j) += tmp(i,k) * tmp(j,k) * cov_z;
            }
        }
    }
    outData[idx]=mat;
}
