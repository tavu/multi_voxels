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

// void saveVoxelsToFile(char *fileName,const Volume volume,const kparams_t &params)
void saveVoxelsToFile(char *fileName,
                      const uint3 &resolution,
                      float vox_size,
                      const tsdfvh::Voxel *voxels)
{    
    std::cout<<"Saving TSDF voxel grid values to disk("<<fileName<<")"<< std::endl;

    std::ofstream outFile(fileName, std::ios::out);
    float dimensions[3];
    dimensions[0]=float(resolution.x);
    dimensions[1]=float(resolution.y);
    dimensions[2]=float(resolution.z);

    outFile<<dimensions[0]<<"\n";
    outFile<<dimensions[1]<<"\n";
    outFile<<dimensions[2]<<"\n";

    float origin[3];
    origin[0]=0.0;
    origin[1]=0.0;
    origin[2]=0.0;

    outFile<<origin[0]<<"\n";
    outFile<<origin[1]<<"\n";
    outFile<<origin[2]<<"\n";

    //assuming cubical voxels   
    outFile<<vox_size<<std::endl;    

    //short2 *voxel_grid_cpu=new short2[volume.getResolution().x*volume.getResolution().y*volume.getResolution().z];

//    uint size=resolution.x*resolution.y*resolution.z;
//    tsdfvh::Voxel *voxel_grid_cpu=new tsdfvh::Voxel[size];

//    cudaMemcpy(voxel_grid_cpu, volume.getVoxelsPtr(),
//               size * sizeof(tsdfvh::Voxel ),
//               cudaMemcpyDeviceToHost);

    uint3 pos;
    for(pos.x=0;pos.x<resolution.x;pos.x++)
    {
        for(pos.y=0;pos.y<resolution.y;pos.y++)
        {
            for(pos.z=0;pos.z<resolution.z;pos.z++)
            {
                uint idx=pos.x + pos.y * resolution.x + pos.z * resolution.x * resolution.y;
                outFile<<voxels[idx].sdf<<'\n';
            }
        }
    }

    outFile.close();

    std::cout<<"Saving done."<<std::endl;
}

