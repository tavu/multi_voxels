#include"volume.h"
#include"marching_cube.h"
#include<iostream>
#include"cutil_math.h"
#include<fstream>
#include "kparams.h"
#include <string>
#include <string.h>


struct out_data
{
    char c[6];
//    float value;
//    char c;
};

void generateTriangles(std::vector<float3>& triangles,  const Volume volume, short2 *hostData)
{
    int3 min=volume.minVoxel();
    int3 max=volume.maxVoxel();
    for(int z=min.z; z<max.z-1; z++)
    {
        for(int y=min.y; y<max.y-1; y++)
        {
            for(int x=min.x;x<max.x-1;x++)
            {
                //Loop over all cubes
                const uint8_t cubeIndex = getCubeIndex(x,y,z,volume, hostData);
                const int* tri = triTable[cubeIndex];

                for(int i=0; i<5; i++)
                {
                    if(tri[3*i]<0)
                        break;

                    float3 p1 = calcPtInterpolate(tri[3*i],x, y, z, volume,hostData);
                    float3 p2 = calcPtInterpolate(tri[3*i+1],x, y, z, volume,hostData);
                    float3 p3 = calcPtInterpolate(tri[3*i+2],x, y, z, volume,hostData);

                    triangles.push_back(p1);
                    triangles.push_back(p2);
                    triangles.push_back(p3);
                }
            }
        }
    }
}



