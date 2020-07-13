%  
function tsdf2mesh(file, outfile)
    headerSize=7;
    M=dlmread(file)';
    tsdfHeader = M(1:headerSize);
    voxelGridDim = tsdfHeader(1:3);
    voxelGridOrigin = tsdfHeader(4:6);
    voxelSize = tsdfHeader(7);
    %truncMargin = tsdfHeader(8);

    tsdf=M(headerSize+1:voxelGridDim(1)*voxelGridDim(2)*voxelGridDim(3)+headerSize);
    % Convert from TSDF to mesh  
    tsdf = reshape(tsdf,[voxelGridDim(1),voxelGridDim(2),voxelGridDim(3)]);
    
    fv = isosurface(tsdf,0);
    points = fv.vertices';
    faces = fv.faces';
    color = uint8(repmat([175;198;233],1,size(points,2)));

    meshPoints(1,:) = voxelGridOrigin(1) + points(2,:)*voxelSize;
    
    % x y axes are swapped from isosurface
    meshPoints(2,:) = voxelGridOrigin(2) + points(1,:)*voxelSize;
    meshPoints(3,:) = voxelGridOrigin(3) + points(3,:)*voxelSize;
    
    % Write header for mesh file
    data = reshape(typecast(reshape(single(meshPoints),1,[]),'uint8'),3*4,[]);
    data = [data; color];
    fid = fopen(outfile,'w');
    fprintf (fid, 'ply\n');
    fprintf (fid, 'format binary_little_endian 1.0\n');
    fprintf (fid, 'element vertex %d\n', size(data,2));
    fprintf (fid, 'property float x\n');
    fprintf (fid, 'property float y\n');
    fprintf (fid, 'property float z\n');
    fprintf (fid, 'property uchar red\n');
    fprintf (fid, 'property uchar green\n');
    fprintf (fid, 'property uchar blue\n');
    fprintf (fid, 'element face %d\n', size(faces,2));
    fprintf (fid, 'property list uchar int vertex_index\n');
    fprintf (fid, 'end_header\n');

    % Write vertices
    fwrite(fid, data,'uint8');

    % Write faces
    faces = faces([3 2 1],:); % reverse the order to get a better normal    
    faces_data = int32(faces-1);
    faces_data = reshape(typecast(reshape(faces_data,1,[]),'uint8'),3*4,[]);
    faces_data = [uint32(ones(1,size(faces,2))*3); faces_data];        
    fwrite(fid, faces_data,'uint8');

    fclose(fid);
end
