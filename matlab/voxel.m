clear;

frame1=504;

dir='/tmp/voxels';
voxelFile = sprintf('%s/f%d_voxels',dir,frame1);
plyFile = sprintf('%s/mesh_%d.ply',dir,frame1);


tsdf2mesh(voxelFile,plyFile);