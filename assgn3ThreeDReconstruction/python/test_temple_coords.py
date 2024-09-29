import numpy as np
import helper as hlp
import skimage.io as io
import submission as sub
import matplotlib.pyplot as plt

import cv2
from submission import eight_point
from submission import essential_matrix
from submission import epipolar_correspondences
from submission import triangulate
import helper
# 1. Load the two temple images and the points from data/some_corresp.npz
temple1=cv2.imread('../data/im1.png')
temple2=cv2.imread('../data/im2.png')
dataPoints=np.load("../data/some_corresp.npz")
# 2. Run eight_point to compute F
M=max(np.shape(temple1))
pts1=dataPoints["pts1"]
pts2=dataPoints["pts2"]
F=eight_point(pts1, pts2, M)

data2 = np.load('../data/intrinsics.npz') 
K1=data2["K1"]
K2=data2["K2"]
E=essential_matrix(F,K1, K2)

# 3. Load points in image 1 from data/temple_coords.npz
dataPoints_temple=np.load("../data/temple_coords.npz")
pts1_corres=dataPoints_temple["pts1"]
# 4. Run epipolar_correspondences to get points in image 2
pts2_corres=epipolar_correspondences(temple1, temple2, F, pts1_corres)
# helper.epipolarMatchGUI(temple1, temple2, F)

# 5. Compute the camera projection matrix P1
#[I|0]
P1_extrinsic=np.eye(3,dtype=int)
P1_extrinsic=np.hstack((P1_extrinsic,np.zeros((3,1),dtype=int)))
P1=K1@P1_extrinsic

# 6. Use camera2 to get 4 camera projection matrices P2
P2s=helper.camera2(E)
#P2s[:,:,k] to access kth camera projection matrix
# 7. Run triangulate using the projection matrices
best_score=0
P2=None
best_k=0
for k in range(4):
    extrinsic_temp=P2s[:,:,k]
    tempP2=K2@extrinsic_temp
    pts3d=triangulate(P1, pts1_corres, tempP2, pts2_corres)
    score=np.count_nonzero(pts3d[:,-1] > 0)
    if(score>best_score):
        best_k=k
        P2=tempP2
        best_score=score

# 8. Figure out the correct P2
T1=P1_extrinsic[:,3:4].reshape(3,)
R1=P1_extrinsic[:,0:3]
R2=P2s[:,:,1][:,0:3]
T2=P2s[:,:,1][:,-1]
np.savez("../data/extrinsics.npz", R1=R1, R2=R2,t1=T1,t2=T2)

pts3d=triangulate(P1, pts1_corres, P2, pts2_corres)


# 9. Scatter plot the correct 3D points
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
x=pts3d[:,0]
y=pts3d[:,1]
z=pts3d[:,2]
ax.scatter(x,y,z,c='b',marker='o',s=20)

#code from piazza post https://piazza.com/class/lluyey2bprp44v/post/lo4qhb8egh52rx
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
 
x_start, x_end = ax.get_xlim()
y_start, y_end = ax.get_ylim()
z_start, z_end = ax.get_zlim()
norm_range = max([x_end - x_start, y_end - y_start, z_end - z_start])
 
ax.set_xlim((x_end + x_start) / 2 - norm_range / 2, (x_end + x_start) / 2 + norm_range / 2)
ax.set_ylim((y_end + y_start) / 2 - norm_range / 2, (y_end + y_start) / 2 + norm_range / 2)
ax.set_zlim((z_end + z_start) / 2 - norm_range / 2, (z_end + z_start) / 2 + norm_range / 2)







plt.show()
# 10. Save the computed extrinsic parameters (R1,R2,t1,t2) to data/extrinsics.npz


