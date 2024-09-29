"""
Homework 5
Submission Functions
"""

# import packages here
import numpy as np
import cv2
import helper
from scipy import signal
"""
Q3.1.1 Eight Point Algorithm
       [I] pts1, points in image 1 (Nx2 matrix)
           pts2, points in image 2 (Nx2 matrix)
           M, scalar value computed as max(H1,W1)
       [O] F, the fundamental matrix (3x3 matrix)
"""
def eight_point(pts1, pts2, M):
    #1. normalize
    pts1_norm=pts1/M
    pts2_norm=pts2/M
    T=np.array([[1/M, 0, 0],[0, 1/M, 0], [0, 0, 1]])
    #2.Construct fundamental matrix F
    N=np.shape(pts1)[0]
    
    A=[]
    for i in range(N):
        xi=pts1_norm[i][0]
        xi_prime=pts2_norm[i][0]
        yi=pts1_norm[i][1]
        yi_prime=pts2_norm[i][1]
        row_i=[xi*xi_prime, xi*yi_prime, xi, yi*xi_prime, yi*yi_prime, yi, xi_prime, yi_prime, 1]
        A+=[row_i]
    A=np.array(A)

    #3. calculate SVD of A
    (U, Sig, Vt)=np.linalg.svd(A)
    F=np.reshape(Vt[-1],(3,3))

    #4. replace 
    (U,Sig,Vt)=np.linalg.svd(F)
    Sig[-1]=0
    F=(U@np.diag(Sig))@Vt

    refinedF=helper.refineF(F, pts1_norm, pts2_norm)
    result=(np.transpose(T)@refinedF)@T

    return result






"""
Q3.1.2 Epipolar Correspondences
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           F, fundamental matrix from image 1 to image 2 (3x3 matrix)
           pts1, points in image 1 (Nx2 matrix)
       [O] pts2, points in image 2 (Nx2 matrix)
"""

def compute_Euclidean(pt1, pt2):
    return np.sum((pt1-pt2)**2)


def compute_best_candidate(candidates, pt1, im1, im2):
    pad_width=5
    pad_dimension=((pad_width, pad_width),
                   (pad_width, pad_width),
                   (0,0))
    
    im1_padded=np.pad(im1, pad_dimension, mode="edge")
    im2_padded=np.pad(im2, pad_dimension, mode="edge")

    #compute the location of x,y in the padded image

    (pt1_x, pt1_y)=(pt1[1]+pad_width, pt1[0]+pad_width)
    pt1_window=im1_padded[pt1_x-pad_width:pt1_x+pad_width+1, 
                          pt1_y-pad_width:pt1_y+pad_width+1,
                          :]
    pt1_window=np.asarray(pt1_window)


    best_distance=-1
    best_candidate_index=-1
    for i in range (len(candidates)):
        #y is col, x is row
        (pt2_x, pt2_y)=(candidates[i][0]+pad_width, candidates[i][1]+pad_width)
        pt2_window=im2_padded[pt2_x-pad_width:pt2_x+pad_width+1,
                              pt2_y-pad_width:pt2_y+pad_width+1,
                              :]
        pt2_window=np.asarray(pt2_window)
        distance=compute_Euclidean(pt1_window, pt2_window)
        distance+=45*(abs(pt2_x-pt1_x)+abs(pt2_y-pt1_y))
        if(i==0 or distance<best_distance):
            best_candidate_index=i
            best_distance=distance
    result=candidates[best_candidate_index]
    return result


def epipolar_correspondences(im1, im2, F, pts1):
    # find candidates
    im2_cols=np.shape(im2)[1]
    result=[]
    for pt1 in pts1:
        #Ex=l'
        epipolar_line=F@np.transpose(np.array([pt1[0], pt1[1], 1]))
        (a,b,c)=(epipolar_line[0],epipolar_line[1], epipolar_line[2])
        candidates=[]
        #calculating candidates
        for x in range(im2_cols):
            #ax+by+c=0, y=(-c-ax)/b
            y=round((-c-a*x)/b)
            #row is y, col is x
            candidates+=[[y,x]]
        best_candidate=compute_best_candidate(candidates, pt1, im1, im2)
        result.append(best_candidate[::-1])
    result=np.array(result)

    return result

        


"""
Q3.1.3 Essential Matrix
       [I] F, the fundamental matrix (3x3 matrix)
           K1, camera matrix 1 (3x3 matrix)
           K2, camera matrix 2 (3x3 matrix)
       [O] E, the essential matrix (3x3 matrix)
"""
def essential_matrix(F, K1, K2):
    #K'^-T E K^-1=F
    #=>E=K'^T F K
    return np.transpose(K2)@F@K1


"""
Q3.1.4 Triangulation
       [I] P1, camera projection matrix 1 (3x4 matrix)
           pts1, points in image 1 (Nx2 matrix)
           P2, camera projection matrix 2 (3x4 matrix)
           pts2, points in image 2 (Nx2 matrix)
       [O] pts3d, 3D points in space (Nx3 matrix)
"""
def triangulate(P1, pts1, P2, pts2):

    result=[]
    for i in range(len(pts1)):
        x=pts1[i][0]
        y=pts1[i][1]
        x_prime=pts2[i][0]
        y_prime=pts2[i][1]
        A=[y*P1[2]-P1[1],
           P1[0] - x*P1[2],
           y_prime*P2[2] - P2[1],
           P2[0]-x_prime*P2[2]]
        A_np=np.array(A)
        U,Sig,Vt=np.linalg.svd(A_np)
        threeD=Vt[-1]
        result.append([threeD[0]/threeD[3], threeD[1]/threeD[3], threeD[2]/threeD[3]])
    return np.array(result)

  

   


"""
Q3.2.1 Image Rectification
       [I] K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] M1 M2, rectification matrices (3x3 matrix)
           K1p K2p, rectified camera matrices (3x3 matrix)
           R1p R2p, rectified rotation matrices (3x3 matrix)
           t1p t2p, rectified translation vectors (3x1 matrix)
"""
def rectify_pair(K1, K2, R1, R2, t1, t2):
    # replace pass by your implementation
    c1=-np.linalg.inv(K1@R1)@(K1@t1)
    c2=-np.linalg.inv(K2@R2)@(K2@t2)

    r1=(c1-c2)/np.linalg.norm(c1-c2)
    r1=r1.reshape((3,))
    r2=np.cross(r1,np.transpose(R1[2,:]))
    r3=np.cross(r2, r1)
    R=np.transpose(np.array([r1,r2,r3]))
 
    R1p=R
    R2p=R

    K1p=K2
    K2p=K2

    t1p=-R@c1
    t2p=-R@c2
 
    M1=(K1p@R)@np.linalg.inv(K1@R1)
    M2=(K2p@R)@np.linalg.inv(K2@R2)

    return(M1,M2,K1p,K2p,R1p,R2p,t1p,t2p)

"""
Q3.2.2 Disparity Map
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           max_disp, scalar maximum disparity value
           win_size, scalar window size value
       [O] dispM, disparity map (H1xW1 matrix)
"""

    
def get_disparity(im1, im2, max_disp, win_size):

    result_map=np.zeros(im1.shape)
    disparity=np.full(im1.shape, np.inf)   
    window=np.ones((win_size,win_size))
    for d in range(max_disp+1):
        img2_transformed=np.roll(im2,d,axis=1)
        convolved=signal.convolve2d((img2_transformed-im1)**2,window,mode="same",boundary='symm')
        result_map[convolved<disparity]=d
        disparity[convolved<disparity]=convolved[convolved<disparity]
    return result_map



"""
Q3.2.3 Depth Map
       [I] dispM, disparity map (H1xW1 matrix)
           K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] depthM, depth map (H1xW1 matrix)
"""
def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    c1=-np.linalg.inv(K1@R1)@(K1@t1)
    c2=-np.linalg.inv(K2@R2)@(K2@t2)
    b=np.linalg.norm(c1-c2)
    f=K1[0][0]
    result=np.zeros_like(dispM)
    result[dispM>0]=b*f/dispM[dispM>0]
    return result
            


"""
Q3.3.1 Camera Matrix Estimation
       [I] x, 2D points (Nx2 matrix)
           X, 3D points (Nx3 matrix)
       [O] P, camera matrix (3x4 matrix)
"""
def estimate_pose(x, X):
    A=[]
    for i in range(np.shape(x)[0]):
        (a,b)=(x[i,0], x[i, 1])
        (a_prime,b_prime, z_prime)=(X[i,0],X[i,1], X[i,2])
        first_row=[-a_prime,-b_prime,-z_prime,-1,0,0,0,0,a*a_prime, a*b_prime, a*z_prime, a]
        second_row=[0, 0, 0, 0, -a_prime, -b_prime, -z_prime, -1, b*a_prime, b*b_prime, b*z_prime, b]
        A.append(first_row)
        A.append(second_row)
    A=np.array(A)
    U,Sigma,VT=np.linalg.svd(A)
    result=VT[-1,:]
    return np.reshape(result,(3,4))


"""
Q3.3.2 Camera Parameter Estimation
       [I] P, camera matrix (3x4 matrix)
       [O] K, camera intrinsics (3x3 matrix)
           R, camera extrinsics rotation (3x3 matrix)
           t, camera extrinsics translation (3x1 matrix)
"""
def estimate_params(P):
    U,Sigma,VT=np.linalg.svd(P)
    c=VT[-1,:]
    new_center=[c[0]/c[3],c[1]/c[3],c[2]/c[3]]
    new_center=np.array(new_center).T
    M=P[:,0:3]
    K,R=np.linalg.qr(M)
   
    t=-M@new_center
    return K,R,t