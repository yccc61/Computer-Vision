import numpy as np
import cv2
import skimage

def computeH(x1, x2):
	#Q3.6
	#Compute the homography between two sets of points
	
	#translate x1 x2 into homogeneous

	A=[]
	for i in range(np.shape(x1)[0]):
		(a,b)=(x2[i,0], x2[i, 1])
		(a_prime,b_prime)=(x1[i,0],x1[i,1])
		first_row =[-a,-b,-1,0 ,0 ,0,a*a_prime,b*a_prime,a_prime]
		second_row=[0,0 ,0 ,-a,-b, -1,a*b_prime,b*b_prime,b_prime]
		A.append(first_row)
		A.append(second_row)
	A=np.array(A)
	U,Sigma,VT=np.linalg.svd(A)
	#should be sorted value
	result=VT[-1,:]
	result=result/result[-1]
	return result.reshape((3,3))



def computeH_norm(x1, x2):
	#Q3.7

	#Compute the centroid of the points
	x1_mean=np.mean(x1,axis=0)
	x2_mean=np.mean(x2,axis=0)
	#Shift the origin of the points to the centroid
	x1=x1-x1_mean
	x2=x2-x2_mean
	#Normalize the points so that the largest distance from the origin is equal to sqrt(2)
	x1_norm=np.sqrt(np.sum(x1**2, axis=1))
	x2_norm=np.sqrt(np.sum(x2**2, axis=1))
	x1_scale=np.sqrt(2)/np.max(x1_norm)
	x2_scale=np.sqrt(2)/np.max(x2_norm)
	x1=x1_scale*x1
	x2=x2_scale*x2

	#Similarity transform 1
	T1=[[x1_scale, 0, -x1_mean[0]*x1_scale],
		[0, x1_scale, -x1_mean[1]*x1_scale],
		[0, 0, 1]]
	T1=np.array(T1)

	#Similarity transform 2
	T2=[[x2_scale,0, -x2_mean[0]*x2_scale],
		[0, x2_scale,-x2_mean[1]*x2_scale],
		[0, 0, 1]]
	T2=np.array(T2)
	#Compute homography
	resultH=computeH(x1, x2)

	
	#Denormalization
	H2to1=np.linalg.inv(T1)@resultH
	H2to1=H2to1@T2

	return H2to1




def computeH_ransac(x1, x2, LoopTimes=400, delta=0.5, k=4):
	#Q3.8
	#Compute the best fitting homography given a list of matching points
	points_count=np.shape(x1)[0]
	inliers=np.zeros(points_count)


	for i in range(LoopTimes):
		sample_index=np.random.choice(points_count, k, replace=False)
		sample_x1=x1[sample_index]
		sample_x2=x2[sample_index]
		currentH=computeH_norm(sample_x1, sample_x2)
		currentInliers=np.zeros(points_count)
		for j in range(points_count):
			result_x1=currentH@ np.array([x2[j][0],x2[j][1],1])
			#convert x1 to heterogeneous coordinates
			hetero_x1=np.array([result_x1[0]/result_x1[2], 
					   			result_x1[1]/result_x1[2]])
			euc_distance=np.linalg.norm(hetero_x1-x1[j])
			if(delta>=euc_distance):
				currentInliers[j]=1
		inlier_count=np.sum(currentInliers)
		if(sum(inliers)<inlier_count):
			inliers=currentInliers

	
	#filter out best x1, x2 with best inliers
	x1=x1[inliers==1]
	x2=x2[inliers==1]
	bestH2to1=computeH_norm(x1,x2)
	return bestH2to1, inliers



def compositeH(H2to1, template, img):
	#Create a composite image after warping the template image on top
	#of the image using the homography

	#Note that the homography we compute is from the image to the template;
	#x_template = H2to1*x_photo
	#For warping the template to the image, we need to invert it.


	H2to1_inv=np.linalg.inv(H2to1)
	
	#Create mask of same size as template
	temp_height=np.shape(template)[0]
	temp_width=np.shape(template)[1]
	temp_shape=(temp_height, temp_width)
	mask=np.ones_like(img)

	#Warp mask by appropriate homography
	warp_mask=skimage.transform.warp(mask, H2to1_inv ,output_shape=temp_shape)
	warp_mask = np.where(warp_mask<=0,1,0)
	
	#Warp template by appropriate homography
	warp_temp=skimage.transform.warp(img,H2to1_inv,output_shape=temp_shape)

	composite_img= (warp_temp*255)+warp_mask*template

	return composite_img





	


