'''
Install opencv:
pip install opencv-python==3.4.2.16
pip install opencv-contrib-python==3.4.2.16
'''

import cv2
import random
import numpy as np
from matplotlib import pyplot as plt
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--UseRANSAC", type=int, default=1 )
parser.add_argument("--image1", type=str,  default='data/myleft.jpg' )
parser.add_argument("--image2", type=str,  default='data/myright.jpg' )
args = parser.parse_args()

print(args)


def FM_by_normalized_8_point(pts1,  pts2):
    #F, _ = cv2.findFundamentalMat(pts1,pts2,  cv2.FM_8POINT )


    # comment out the above line of code.

    # Your task is to implement the algorithm by yourself.
    # Do NOT copy&paste any online implementation.

    # F:  fundmental matrix
    m = np.zeros((pts1.shape[0],9))

    #Normalize inputs
    t_1, pts1 = normalize_points(pts1)
    t_2, pts2 = normalize_points(pts2)

    #Construct coeff matrix
    for i in range(pts1.shape[0]):
        m[i,0] = pts1[i,0] * pts2[i,0]
        m[i,1] = pts1[i,1] * pts2[i,0]
        m[i,2] = pts2[i,0]
        m[i,3] = pts1[i,0] * pts2[i,1]
        m[i,4] = pts1[i,1] * pts2[i,1]
        m[i,5] = pts2[i,1]
        m[i,6] = pts1[i,0]
        m[i,7] = pts1[i,1]
        m[i,8] = 1

    #Compute F
    #find_eig = m.T@ m
    #u,v = np.linalg.eig(find_eig)

    #U, S, V = np.linalg.svd(m)
    #print(V[-1])
    #F = V[-1].reshape(3, 3)
    u, sigma, vT= svd_scratch(m)
    #vec = vT[:,np.where(min(sigma)[0])]
    vec = -vT[-1]
    #print(vec)
    F_norm = vec.reshape((3,3))
    F = t_2.T @ F_norm @ t_1

    #v,d = np.linalg.eig(F)
    s,v,d = np.linalg.svd(F)
    p,q,r = svd_scratch(F)

    v[2] = 0
    F = s @ np.diag(v) @ d
    #F = np.dot(s, np.dot(np.diag(v), d))
    #print(m[0])

    return F/F[2,2]



def svd_scratch(m):
    mmT = m @ m.T
    mTm = m.T @ m

    x,y = np.linalg.eig(mmT)
    idx_1 = np.argsort(x)

    x= x[idx_1][::-1]
    y = y[idx_1][::-1]

    sing_vals = np.sqrt(np.abs(x))

    p,q = np.linalg.eig(mTm)
    idx_2 = np.argsort(p)

    p= p[idx_2][::-1]
    vT = q.T
    vT = vT[idx_2][::-1]

    u = y
    sigma = np.diag(sing_vals)
    if sigma.shape[0] < u.shape[1]:
        sigma = np.vstack((sigma,np.zeros((u.shape[1]-sigma.shape[0], sigma.shape[1]))))

    if sigma.shape[1] < vT.shape[0]:
        sigma = np.hstack((sigma,np.zeros((vT.shape[0]-sigma.shape[1], sigma.shape[0])).reshape(-1,1)))

    return u, sigma, vT
def normalize_points(points):

    #Find centroid of points
    mean_x, mean_y = np.mean(points, axis = 0)

    #Find mean distance from centroid
    x_dist = points[:,0] - mean_x
    y_dist = points[:,1] - mean_y
    dist = np.sqrt(x_dist**2 + y_dist**2)
    mean_dist = np.sum(dist)/len(dist)

    #Formulate normalizing matrix
    m = np.array([[np.sqrt(2)/mean_dist, 0, -mean_x * (np.sqrt(2)/mean_dist)],
                 [0, np.sqrt(2)/mean_dist, -mean_y * np.sqrt(2)/mean_dist],
                 [0, 0, 1]])

    points = np.hstack((points,np.ones(points.shape[0]).reshape(-1,1)))
    new_points = m @ points.T

    return m, new_points[0:2,:].T

def FM_by_RANSAC(pts1,  pts2):
    #F, mask = cv2.findFundamentalMat(pts1,pts2,  cv2.FM_RANSAC )
    # comment out the above line of code.

    # Your task is to implement the algorithm by yourself.
    # Do NOT copy&paste any online implementation.

    # F:  fundmental matrix
    # mask:   whetheter the points are inliers

    #Sample randomly and compute F on random samples
    inlier_prev = 0
    threshold = .02

    for i in range(500):

        idx = random.sample(range(pts1.shape[0]),8)
        F = FM_by_normalized_8_point(pts1[idx], pts2[idx])
        if type(F) != type(None):

            inlier_cur = 0
            mask_new = np.zeros((pts1.shape[0],1))
        #compute inliers
            for j in range(pts1.shape[0]):
                dist = np.hstack((pts1[j],np.array([1]))).reshape((1,3)) @ F @ np.hstack((pts2[j],np.array([1]))).reshape((3,1))
                if np.abs(dist) < threshold:
                    inlier_cur += 1
                    mask_new[j] = 1

            if inlier_cur > inlier_prev:
                F_best = F
                inlier_prev = inlier_cur
                mask_best = mask_new.copy()


        if inlier_cur == pts1.shape[0]:
            break

    return  F_best, mask_best






img1 = cv2.imread(args.image1,0)
img2 = cv2.imread(args.image2,0)

sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

good = []
pts1 = []
pts2 = []

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)


pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

F = None
if args.UseRANSAC:
    F,  mask = FM_by_RANSAC(pts1,  pts2)
    # We select only inlier points
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]
else:
    F = FM_by_normalized_8_point(pts1,  pts2)


def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2


# Find epilines corresponding to points in second image,  and draw the lines on first image
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,  F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img6)
plt.show()

# Find epilines corresponding to points in first image, and draw the lines on second image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
plt.subplot(121),plt.imshow(img4)
plt.subplot(122),plt.imshow(img3)
plt.show()
