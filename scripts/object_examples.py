from gibson2.objects.ycb_object import YCBObject
from gibson2.objects.articulated_object import ArticulatedObject
import gibson2
import os
import sys
import pybullet as p
import pybullet_data
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
from shapely.geometry import box, Polygon

def iou_coverage(bb, image_bb):
    bb_edgepoints=bb_to_edgepoints(bb)
    im_edgepoints=bb_to_edgepoints(image_bb)
    bb_shape=Polygon(bb_edgepoints)
    im_shape=Polygon(im_edgepoints)

    intersection=bb_shape.intersection(im_shape).area
    union=bb_shape.union(im_shape).area
    iou=intersection/union
    return iou

def bb_to_edgepoints(bb):
    bb_min_x = min(bb[0][0], bb[1][0])
    bb_max_x = max(bb[0][0], bb[1][0])
    bb_min_y = min(bb[0][1], bb[1][1])
    bb_max_y = max(bb[0][1], bb[1][1])
    return [[bb_min_x, bb_min_y], [bb_min_x, bb_max_y], [bb_max_x, bb_max_y], [bb_max_x, bb_min_y]]





def world_to_img(world_coord, projectionMatrix, viewMatrix, imwidth, imheight):
    fx=0.9
    fy=0.9
    own_K=np.array([[fx, 0, 0, 0],
                   [0, fy, 0, 0],
                   [0, 0, 1, 0]])
    K = np.asarray(projectionMatrix).reshape(4, 4).T#[:3, :4]
    K = np.delete(K, 2, 0)
    K[2][2]=1
    #K=own_K
    Rt = np.asarray(viewMatrix).reshape(4, 4).T
    print("instrinsic K:\n", K)
    print("extrinsic Rt:\n", Rt)

    x_im_coord_hom = (K @ Rt) @ np.concatenate((world_coord, np.array([1])))
    x_im_coord_2d = np.array([x_im_coord_hom[0] / x_im_coord_hom[2], x_im_coord_hom[1] / x_im_coord_hom[2]])
    """coord_to_pixel_scale = 1#imwidth
    x_im_pixel = np.multiply(x_im_coord_2d, coord_to_pixel_scale)
    #x_im_pixel = np.multiply(x_im_pixel, np.array([1, -1]))
    return x_im_pixel#np.add(x_im_pixel, np.array([imwidth / 2, imheight / 2]))"""
    print("x im coord 2d",x_im_coord_2d)
    sc=(x_im_coord_2d+1)/2
    print("sc:",sc)
    scaled=sc*imwidth
    print("scaled:",scaled)
    return scaled#[-scaled[0], scaled[1]-400]

    #return norm_pix#np.array([(imwidth/2)-norm_pix[0],(imheight/2)-norm_pix[1]])#np.array([200+norm_pix[0],200-norm_pix[1]])#np.array([(imwidth/2+norm_pix[0]/2),(imheight/2)-norm_pix[1]/2])#-np.array([imwidth/2,imheight/2])


def drawAABB(aabb):
    aabbMin = aabb[0]
    aabbMax = aabb[1]
    f = [aabbMin[0], aabbMin[1], aabbMin[2]]
    t = [aabbMax[0], aabbMin[1], aabbMin[2]]
    p.addUserDebugLine(f, t, [1, 0, 0])
    f = [aabbMin[0], aabbMin[1], aabbMin[2]]
    t = [aabbMin[0], aabbMax[1], aabbMin[2]]
    p.addUserDebugLine(f, t, [0, 1, 0])
    f = [aabbMin[0], aabbMin[1], aabbMin[2]]
    t = [aabbMin[0], aabbMin[1], aabbMax[2]]
    p.addUserDebugLine(f, t, [0, 0, 1])

    f = [aabbMin[0], aabbMin[1], aabbMax[2]]
    t = [aabbMin[0], aabbMax[1], aabbMax[2]]
    p.addUserDebugLine(f, t, [1, 1, 1])

    f = [aabbMin[0], aabbMin[1], aabbMax[2]]
    t = [aabbMax[0], aabbMin[1], aabbMax[2]]
    p.addUserDebugLine(f, t, [1, 1, 1])

    f = [aabbMax[0], aabbMin[1], aabbMin[2]]
    t = [aabbMax[0], aabbMin[1], aabbMax[2]]
    p.addUserDebugLine(f, t, [1, 1, 1])

    f = [aabbMax[0], aabbMin[1], aabbMin[2]]
    t = [aabbMax[0], aabbMax[1], aabbMin[2]]
    p.addUserDebugLine(f, t, [1, 1, 1])

    f = [aabbMax[0], aabbMax[1], aabbMin[2]]
    t = [aabbMin[0], aabbMax[1], aabbMin[2]]
    p.addUserDebugLine(f, t, [1, 1, 1])

    f = [aabbMin[0], aabbMax[1], aabbMin[2]]
    t = [aabbMin[0], aabbMax[1], aabbMax[2]]
    p.addUserDebugLine(f, t, [1, 1, 1])

    f = [aabbMax[0], aabbMax[1], aabbMax[2]]
    t = [aabbMin[0], aabbMax[1], aabbMax[2]]
    p.addUserDebugLine(f, t, [1.0, 0.5, 0.5])
    f = [aabbMax[0], aabbMax[1], aabbMax[2]]
    t = [aabbMax[0], aabbMin[1], aabbMax[2]]
    p.addUserDebugLine(f, t, [1, 1, 1])
    f = [aabbMax[0], aabbMax[1], aabbMax[2]]
    t = [aabbMax[0], aabbMax[1], aabbMin[2]]
    p.addUserDebugLine(f, t, [1, 1, 1])

def main(urdf_input):
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(1./240.)

    floor = os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
    p.loadMJCF(floor)

    obj = p.loadURDF(urdf_input)

    #Plane Bounding Box
    plane_bb = p.getAABB(obj, linkIndex=0)
    #drawAABB(plane_bb)

    #Handle Bounding Box
    handle_bb = p.getAABB(obj, linkIndex=1)
    #drawAABB(handle_bb)

    #Complete Door Bounding Box
    all_bb_points = [plane_bb[0], plane_bb[1], handle_bb[0], handle_bb[1]]
    complete_bb = [np.min(all_bb_points,axis=0),np.max(all_bb_points,axis=0)]
    #drawAABB(complete_bb)

    #Get rotation axis
    handle_center_y = (handle_bb[0][1]+handle_bb[1][1])/2
    plane_center_y = (plane_bb[0][1]+plane_bb[1][1])/2
    if handle_center_y>plane_center_y:
        axis=[[(plane_bb[0][0]+plane_bb[1][0])/2,plane_bb[0][1],plane_bb[0][2]],
              [(plane_bb[0][0]+plane_bb[1][0])/2, plane_bb[0][1], plane_bb[1][2]]]
    else:
        axis=[[(plane_bb[0][0]+plane_bb[1][0])/2, plane_bb[1][1], plane_bb[0][2]],
              [(plane_bb[0][0]+plane_bb[1][0])/2, plane_bb[1][1], plane_bb[1][2]]]
    drawAABB(axis)

    """eye_xs = np.linspace(-3, -1, 4)
    eye_ys = np.linspace(-2, 2, 4)
    eye_zs = np.linspace(0.5, 2.5, 4)

    tar_ys = np.linspace(-1, 1, 4)
    tar_zs = np.linspace(0, 2, 4)"""
    eye_xs = np.linspace(-3, -3, 1)
    eye_ys = np.linspace(2, 2, 1)
    eye_zs = np.linspace(2.5, 2.5, 1)

    tar_ys = np.linspace(0.0, 1.0, 1)
    tar_zs = np.linspace(0.5, 0.5, 1)

    for eye_x in eye_xs:
        for eye_y in eye_ys:
            for eye_z in eye_zs:
                for tar_y in tar_ys:
                    for tar_z in tar_zs:
                        viewMatrix = p.computeViewMatrix(
                            cameraEyePosition=[eye_x,eye_y,eye_z],
                            cameraTargetPosition=[0, tar_y, tar_z],
                            cameraUpVector=[0, 0, 1])

                        projectionMatrix = p.computeProjectionMatrixFOV(
                            fov=45.0,
                            aspect=1.0,
                            nearVal=0.1,
                            farVal=20.1)

                        width, height, rgbImg, depthImg, segImg = p.getCameraImage(
                            width=400,
                            height=400,
                            viewMatrix=viewMatrix,
                            projectionMatrix=projectionMatrix)
                        print("IOU:",iou_coverage(bb=[[-width,-height],[width, height]], image_bb=[[0,0],[width,height]]))

                        imcoord=world_to_img(world_coord=[0,0.3,1],projectionMatrix=projectionMatrix, viewMatrix=viewMatrix, imwidth=width, imheight=height)
                        print("imcoord:",imcoord)
                        #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                        #cv2.resizeWindow('image', width, height)
                        marked_rgbImg=cv2.circle(rgbImg, (int(imcoord[0]),int(imcoord[1])), radius=1, color=(0,0,255), thickness=10)
                        #cv2.imshow('image', marked_rgbImg)
                        #cv2.waitKey(0)
                        #cv2.destroyAllWindows()
                        #plt.imshow(rgbImg)
                        #plt.scatter(imcoord[0],imcoord[1])
                        #plt.pause(1.0)
                        #plt.clf()

                        #plt.imsave(fname="rgb_"+str(y)+".png", arr=rgbImg)
                        #plt.imsave(fname="dep_" + str(y) + ".png", arr=depthImg)
                        #pos_rot = p.getBasePositionAndOrientation(obj)
    #plt.show()

    for _ in range(24000):  # at least 100 seconds
        p.stepSimulation()
        time.sleep(1./240.)

    p.disconnect()




if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("No urdf specified. Aborting!")
        exit(-1)
    urdf_input = sys.argv[1]

    main(urdf_input)
