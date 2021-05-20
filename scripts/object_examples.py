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

    # urdf_path = 'urdf_xacro/'
    # urdf_file = urdf_path + urdf_input
    # door = os.path.join(gibson2.assets_path, 'models/cabinet/door.urdf')
    # door = 'door.urdf'
    # handle = 'door_handle.urdf'
    # obj_door = ArticulatedObject(filename=door)
    # obj_door.load()
    # obj_door.set_position([0, 0, 0])

    # obj = ArticulatedObject(filename=urdf_input)
    # obj.load()
    # obj.set_position([0, 0, 0])

    pos = np.arange(0, 6)
    z_pos = np.random.normal(1, 0)
    obj = p.loadURDF(urdf_input)

    #Boudning box
    door_depth = 0.04
    door_width = 0.75
    door_height = 2.0
    door_center = np.array([0.0, 0.0, 1.0])
    bb_min_w = np.subtract(door_center, 0.5*np.array([2*door_depth, door_width, door_height]))
    bb_max_w = np.add(door_center, 0.5 * np.array([0*door_depth, door_width, door_height]))
    print("BB MIN: ", bb_min_w, "BB MAX", bb_max_w)
    drawAABB([bb_min_w, bb_max_w])
    """R=np.array([3,2,3])
    camera_origin=np.array([1,1,1])
    print("shape:",np.shape(R[0]))
    t=np.multiply(-R, camera_origin)
    print("t:###################################")
    print(t)
    extr=np.array([[R[0],t[0]],
                   [R[1],t[1]],
                   [R[2],t[2]],
                   [0,1]])
    print("extr:", extr)
    world_coordinate=np.array([1,2,3,4]).T
    #bb_min_c=np.multiply(extr.T,world_coordinate.T)
    b#b_min_c=np.vdot(extr, world_coordinate)

    print("bb min camera!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1")
    print(bb_min_c)"""




    ys = np.linspace(-2, 2, 20)
    #count = 0
    #while True:
    for y in ys:

        # x = np.random.choice(pos)
        # y = np.random.choice(pos)
        # z = np.random.choice(pos)
        #y = ys[count % len(ys)]
        #count += 1
        
        viewMatrix = p.computeViewMatrix(
            cameraEyePosition=[1.5,y,1],
            cameraTargetPosition=[0, 0, 1],
            cameraUpVector=[0, 0, 1])
        print("viewmatrix:",np.asarray(viewMatrix).reshape(4,4).T)

        projectionMatrix = p.computeProjectionMatrixFOV(
            fov=45.0,
            aspect=1.0,
            nearVal=0.1,
            farVal=20.1)
        print("projmatrix:",projectionMatrix)

        width, height, rgbImg, depthImg, segImg = p.getCameraImage(
            width=400, 
            height=400,
            viewMatrix=viewMatrix,
            projectionMatrix=projectionMatrix)
        
        #Convert bb to image coordinates:
        K=np.asarray(projectionMatrix).reshape(4,4)[:3,:4]
        Rt=np.asarray(viewMatrix).reshape(4,4).T#.T[:3,:4]
        #x_I=np.multiply(np.multiply(K,Rt),np.array([1,1,1,1]))
        x_I_hom = (K @ Rt) @ np.concatenate((bb_min_w,np.array([1])))
        print("xi hom:",x_I_hom)
        xi_2d=np.array([x_I_hom[0]/x_I_hom[2],x_I_hom[1]/x_I_hom[2]])
        print("xi 2d non hom:", xi_2d)
        print("K: ",K)

        plt.imshow(rgbImg)
        plt.scatter(1,1)
        plt.show()





        print('rgb = ', rgbImg.shape, ', d = ', depthImg.shape, segImg.shape)
        #plt.imsave(fname="rgb_"+str(y)+".png", arr=rgbImg)
        #plt.imsave(fname="dep_" + str(y) + ".png", arr=depthImg)
        pos_rot = p.getBasePositionAndOrientation(obj)
        print(pos_rot)

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
