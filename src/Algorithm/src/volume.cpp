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

// void saveVoxelsToFile(char *fileName,const Volume volume,const kparams_t &params)
void saveVoxelsToFile(char *fileName,const Volume volume)
{
    //TODO this function needs cleanup and speedup
    std::cout<<"Saving TSDF voxel grid values to disk("<<fileName<<")"<< std::endl;

    std::ofstream outFile(fileName, std::ios::out);
    float dimensions[3];
    dimensions[0]=float(volume.getResolution().x);
    dimensions[1]=float(volume.getResolution().y);
    dimensions[2]=float(volume.getResolution().z);

    outFile<<dimensions[0]<<std::endl;
    outFile<<dimensions[1]<<std::endl;
    outFile<<dimensions[2]<<std::endl;

    float origin[3];
    origin[0]=0.0;
    origin[1]=0.0;
    origin[2]=0.0;

    outFile<<origin[0]<<std::endl;
    outFile<<origin[1]<<std::endl;
    outFile<<origin[2]<<std::endl;

    //assuming cubical voxels
    float vox_size=volume.getVoxelSize().x;
    //float vox_size=float(params.volume_size.x)/float(params.volume_resolution.x);
    outFile<<vox_size<<std::endl;
    //outFile<<params.mu<<std::endl;

    short2 *voxel_grid_cpu=new short2[volume.getResolution().x*volume.getResolution().y*volume.getResolution().z];

    cudaMemcpy(voxel_grid_cpu, volume.getDataPtr(),
                   volume.getResolution().x*volume.getResolution().y*volume.getResolution().z* sizeof(short2),
                   cudaMemcpyDeviceToHost);

    //for(int i=0;i<params.volume_resolution.x*params.volume_resolution.y*params.volume_resolution.z;i++)

    uint3 pos;
    for(pos.x=0;pos.x<volume.getResolution().x;pos.x++)
    {
        for(pos.y=0;pos.y<volume.getResolution().y;pos.y++)
        {
            for(pos.z=0;pos.z<volume.getResolution().z;pos.z++)
            {
                uint arrayPos=volume.getIdx(pos);
                short2 data=voxel_grid_cpu[arrayPos];
                float value=float(data.x)/32766.0f;
                outFile<<value<<'\n';
            }
        }
    }

    outFile.close();

    std::cout<<"Saving done."<<std::endl;
    delete [] voxel_grid_cpu;
}

