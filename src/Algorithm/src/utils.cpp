#include"utils.h"

#include<eigen3/Eigen/Dense>


struct timespec tick_clockData;
struct timespec tock_clockData;
bool print_kernel_timing = false;

sMatrix4 T_B_P(0,-1,  0 ,0,
               0, 0, -1, 0,
               1, 0,  0, 0,
               0, 0,  0, 1 );

sMatrix4 invT_B_P=inverse(T_B_P);

float2 checkPoseErr(sMatrix4 p1,sMatrix4 p2)
{
    float2 ret;
    sMatrix3 r1,r2;

    float3 tr1=make_float3(p1(0,3),p1(1,3),p1(2,3));
    float3 tr2=make_float3(p2(0,3),p2(1,3),p2(2,3));

    tr1=tr1-tr2;
    ret.x=l2(tr1);

    for(int i=0;i<3;i++)
    {
        for(int j=0;j<3;j++)
        {
            r1(i,j)=p1(i,j);
            r2(i,j)=p2(i,j);
        }
    }
    r1=r1*transpose(r2);
    float3 f=logMap(r1);

    ret.y=l2(f);
    return ret;
}

inline Eigen::MatrixXd toEigen(const sMatrix4 &mat)
{
    Eigen::MatrixXd ret(4,4);
    for(int i=0;i<4;i++)
    {
        for(int j=0;j<4;j++)
        {
            ret(i,j)=mat(i,j);
        }
    }
    return ret;
}

inline Eigen::MatrixXd toEigen(const sMatrix3 &mat)
{
    Eigen::MatrixXd ret(3,3);
    for(int i=0;i<3;i++)
    {
        for(int j=0;j<3;j++)
        {
            ret(i,j)=mat(i,j);
        }
    }
    return ret;
}

inline Eigen::MatrixXd toEigen(const sMatrix6 &mat)
{
    Eigen::MatrixXd ret(6,6);
    for(int i=0;i<6;i++)
    {
        for(int j=0;j<6;j++)
        {
            ret(i,j)=mat(i,j);
        }
    }
    return ret;
}

inline sMatrix4 fromEigen4(const Eigen::MatrixXd &mat)
{
    sMatrix4 ret;
    for(int i=0;i<4;i++)
    {
        for(int j=0;j<4;j++)
        {
            ret(i,j)=mat(i,j);
        }
    }
    return ret;
}

inline sMatrix6 fromEigen6(const Eigen::MatrixXd &mat)
{
    sMatrix6 ret;
    for(int i=0;i<6;i++)
    {
        for(int j=0;j<6;j++)
        {
            ret(i,j)=mat(i,j);
        }
    }
    return ret;
}

inline sMatrix3 fromEigen3(const Eigen::MatrixXd &mat)
{
    sMatrix3 ret;
    for(int i=0;i<3;i++)
    {
        for(int j=0;j<3;j++)
        {
            ret(i,j)=mat(i,j);
        }
    }
    return ret;
}

sMatrix4 inverse(const sMatrix4 & A)
{
//    Eigen::MatrixXd mat=toEigen(A);
//    mat=mat.inverse();
//    return fromEigen4(mat);

    static TooN::Matrix<4, 4, float> I = TooN::Identity;
    TooN::Matrix<4, 4, float> temp = TooN::wrapMatrix<4, 4>(&A.data[0].x);
    sMatrix4 R;
    TooN::wrapMatrix<4, 4>(&R.data[0].x) = TooN::gaussian_elimination(temp, I);
    return R;

}

sMatrix6 inverse(const sMatrix6 & A)
{
//    Eigen::MatrixXd mat=toEigen(A);
//    mat=mat.inverse();
//    return fromEigen6(mat);

    static TooN::Matrix<6, 6, float> I = TooN::Identity;
    TooN::Matrix<6, 6, float> temp = TooN::wrapMatrix<6, 6>(&A.data[0]);
    sMatrix6 R;
    TooN::wrapMatrix<6, 6>(&R.data[0]) = TooN::gaussian_elimination(temp, I);
    return R;

}
