from gibson2.objects.ycb_object import YCBObject
from gibson2.objects.articulated_object import ArticulatedObject
import gibson2
import os
import sys
import pybullet as p
import pybullet_data
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
from shapely.geometry import box, Polygon, LineString, Point
import shapely
import random

# Clips bb to image dims to ensure stability and compute iou coverage
# between bb(img coordinates) and image with helper function "iou_coverage"
def check_coverage(bb, imwidth, imheight):
    im_boundary = np.copy(bb)
    im_boundary[0][0] = np.clip(im_boundary[0][0], 0, imwidth)
    im_boundary[1][0] = np.clip(im_boundary[1][0], 0, imwidth)
    im_boundary[0][1] = np.clip(im_boundary[0][1], 0, imheight)
    im_boundary[1][1] = np.clip(im_boundary[1][1], 0, imheight)

    iou = iou_coverage(bb, im_boundary)
    return iou

# Checks if at least one of the two axis endpoints is in the image
def rotation_axis_covered(axis_img, imwidth, imheight):
    if axis_img[0][0]<imwidth and axis_img[0][0]>0 and axis_img[0][1]<imheight and axis_img[0][1]>0:
        return True
    if axis_img[1][0]<imwidth and axis_img[1][0]>0 and axis_img[1][1]<imheight and axis_img[1][1]>0:
        return True
    return False

# Applying image weighted metric(edge coverage, plane coverage, handle coverage) 
# and comparing the resulting score to min_score. Return True is score is >= min_score
def good_image(plane_bb_img, handle_bb_img, axis_img, imwidth, imheight, min_ec=0.5, ec_weight=1.0, iou_weight=1.0, min_score=0.5):
    ec=edge_coverage(plane_bb_img, imwidth, imheight)
    iou_plane=check_coverage(plane_bb_img, imwidth, imheight)
    handle_covered=check_coverage(handle_bb_img, imwidth, imheight)>0.9
    axis_covered=rotation_axis_covered(axis_img=axis_img, imwidth=imwidth, imheight=imheight)
    score=(ec*ec_weight+iou_plane*iou_weight)/(ec_weight+iou_weight) if ec>=min_ec and axis_covered and handle_covered else 0
    return score>=min_score

# Gets object bounding box in image coordinates (bb), image dimensions (imwidth, imheight)
# and returns percentage (0-1) of the iou coverage between image and given bb.
def iou_coverage(bb, image_bb):
    bb_edgepoints = bb_to_edgepoints(bb)
    im_edgepoints = bb_to_edgepoints(image_bb)
    bb_shape = Polygon(bb_edgepoints)
    im_shape = Polygon(im_edgepoints)

    intersection = bb_shape.intersection(im_shape).area
    union = bb_shape.union(im_shape).area
    if union>0:
        iou = intersection / union
    else: iou=0
    return iou

# Check which percentage of edges of bb are covered in the image
def edge_coverage(bb, imwidth, imheight):
    edges = bb_to_edgepoints(bb)
    edge_coverage=len(edges)
    for e in edges:
        if e[0]>imwidth or e[0]<0:
            edge_coverage-=1
        elif e[1]>imheight or e[1]<0:
            edge_coverage-=1
    return edge_coverage/len(edges)

# Helperfunction to extract edge points of bb
def bb_to_edgepoints(bb):
    bb_min_x = min(bb[0][0], bb[1][0])
    bb_max_x = max(bb[0][0], bb[1][0])
    bb_min_y = min(bb[0][1], bb[1][1])
    bb_max_y = max(bb[0][1], bb[1][1])
    return [[bb_min_x, bb_min_y], [bb_min_x, bb_max_y], [bb_max_x, bb_max_y], [bb_max_x, bb_min_y]]

# Applying transform from 3D world coorinate to 2D img coordinate
# projectionMatrix and viewMatrix are of the type of PyBullet derived from OpenGL
# and can be compute with p.computeProjectionMatrixFOV and p.computeViewMatrix
def world_to_img(world_coord, projectionMatrix, viewMatrix, imwidth, imheight):
    K_x = np.asarray(projectionMatrix).reshape(4, 4).T  # [:3, :4]
    K_x = np.delete(K_x, 2, 0)
    K_x[2][2] = -1
    Rt_x = np.asarray(viewMatrix).reshape(4, 4).T
    x_im_coord_hom = (K_x @ Rt_x) @ np.concatenate((world_coord, np.array([1])))
    x_im_coord_2d = np.array([x_im_coord_hom[0] / x_im_coord_hom[2], x_im_coord_hom[1] / x_im_coord_hom[2]])
    x_sc = (x_im_coord_2d + 1) / 2
    x_scaled = x_sc * imwidth

    K_y = np.asarray(projectionMatrix).reshape(4, 4).T  # [:3, :4]
    K_y = np.delete(K_y, 2, 0)
    K_y[2][2] = 1
    Rt_y = np.asarray(viewMatrix).reshape(4, 4).T
    y_im_coord_hom = (K_y @ Rt_y) @ np.concatenate((world_coord, np.array([1])))
    y_im_coord_2d = np.array([y_im_coord_hom[0] / y_im_coord_hom[2], y_im_coord_hom[1] / y_im_coord_hom[2]])
    y_sc = (y_im_coord_2d + 1) / 2
    y_scaled = y_sc * imheight

    return [int(x_scaled[0]),int(y_scaled[1])]

# Function for drawing bounding box in PB GUI, can be removed 
# but maybe it's useful to visualize bbs for debugging
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

def find_files(filename, search_path):
    result = []
    # Wlaking top-down from the root
    for root, dir, files in os.walk(search_path):
        if filename in files:
            result.append(os.path.join(root, filename))
    return result


def get_json_data(file_path=None):
    if file_path:
        json_path = os.path.abspath(file_path)
    else:
        json_path = os.path.abspath(find_files('door_features_template.json', '../')[0])
    with open(json_path, ) as file:
        data = json.load(file)
    return data

# Saving RGBD images + corresponding json file with the features
# (bb_door, bb_handle, rotation(axis) and axis_is_right)
def generate_datapoint(file_name, bb_door, bb_handle, rotation, axis_is_right, json_path=None, **kwargs):
    '''
        Function: saves images and its corresponding features (in a json-file)
        Input:
            bb_door:   min/max-Points of 2D door bounding-box
            bb_handle: min/max-Points of 2D handle bounding-box
    '''

    output_path = '../data/train_data/'

    data = get_json_data(json_path)
    data['object']['min'] = bb_door[0]
    data['object']['max'] = bb_door[1]
    data['handle']['min'] = bb_handle[0]
    data['handle']['max'] = bb_handle[1]
    data['axis'] = rotation
    data['axis_is_right'] = axis_is_right

    # save images
    if 'rgb_img' in kwargs.keys():
        rgb_out = os.path.join(output_path, 'images/') + file_name.replace('.FORMAT', '_rgb.jpg')  # or jpg
        #plt.imsave(rgb_out,kwargs['rgb_img'], format='jpg', quality=90)
        # cv2 imwrite takes quality parameter in position 2, >94 might be unstable according to cv2 doc
        cv2.imwrite(rgb_out, cv2.cvtColor(kwargs['rgb_img'],cv2.COLOR_BGR2RGB), [int(cv2.IMWRITE_JPEG_QUALITY), 94])

    if 'depth_img' in kwargs.keys():
        depth_out = os.path.join(output_path, 'images/') + file_name.replace('.FORMAT', '_depth.png')
        plt.imsave(depth_out,kwargs['depth_img'], format='png')
    if 'seg_img' in kwargs.keys():
        seg_out = os.path.join(output_path, 'images/') + file_name.replace('.FORMAT', '_seg.png')
        plt.imsave(seg_out,kwargs['seg_img'], format='png')

    # save features in json
    json_out = os.path.join(output_path, 'features/') + file_name.replace('.FORMAT', '.json')
    with open(json_out, 'w') as f:
        json.dump(data, f, indent=4)

# Extracts rotation axis from plane_bb and handle_bb
# It's comparing handle-and plane center and determining if 
# handle is right or left side. Then return axis_is_right boolean
# and corner point of plane bb on the opposite side of handle as axis
def get_rotation_axis(plane_bb, handle_bb):
    handle_center_y = (handle_bb[0][1] + handle_bb[1][1]) / 2
    plane_center_y = (plane_bb[0][1] + plane_bb[1][1]) / 2
    axis_is_right=0 if handle_center_y > plane_center_y else 1
    if axis_is_right==0:
        axis = [[(plane_bb[0][0] + plane_bb[1][0]) / 2, plane_bb[0][1], plane_bb[0][2]],
                [(plane_bb[0][0] + plane_bb[1][0]) / 2, plane_bb[0][1], plane_bb[1][2]]]
    else:
        axis = [[(plane_bb[0][0] + plane_bb[1][0]) / 2, plane_bb[1][1], plane_bb[0][2]],
                [(plane_bb[0][0] + plane_bb[1][0]) / 2, plane_bb[1][1], plane_bb[1][2]]]
    return axis_is_right, axis

# Uses p.setjointMotorControl to move door to desired opening angle, 
# door-urdf has to contain door joint at index 0
def set_door_angle(goal_angle, obj):
    p.setJointMotorControl2(obj, 0, p.POSITION_CONTROL, goal_angle, force=999)
    while p.getJointState(obj, 0)[0] < goal_angle - 0.001:
        p.stepSimulation()
        time.sleep(1. / 240.)

# Going along grid of follwoing camera parameters:
# eye_xs, eye_ys, eye_zs: positions of camera in world cooridinates
# tar_ys, tar_zs: view targets of camera (where to look at)=> camera orientation
# door_angles: Determines how far to open the door
# During iterating the grid image quality is checked and if good creating datapoint(img+feature json)
def generate_data_imgs(obj, urdf_input, env_input, eye_xs, eye_ys, eye_zs, tar_ys, tar_zs, door_angles):
    for door_angle in door_angles:
        for eye_x in eye_xs:
            eye_x = random.uniform(eye_x - abs(eye_x / 20), eye_x + abs(eye_x / 20))
            for eye_y in eye_ys:
                eye_y = random.uniform(eye_y - abs(eye_y / 20), eye_y + abs(eye_y / 20))
                for eye_z in eye_zs:
                    eye_x = random.uniform(eye_z - abs(eye_z / 20), eye_z + abs(eye_z / 20))
                    for tar_y in tar_ys:
                        tar_y = random.uniform(tar_y - abs(tar_y / 20), tar_y + abs(tar_y / 20))
                        for tar_z in tar_zs:
                            tar_z = random.uniform(tar_z - abs(tar_z / 20), tar_z + abs(tar_z / 20))
                            viewMatrix = p.computeViewMatrix(
                                cameraEyePosition=[eye_x, tar_y, eye_z],
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
                            set_door_angle(door_angle,obj)
                            plane_bb=p.getAABB(obj, linkIndex=0)
                            handle_bb=p.getAABB(obj, linkIndex=1)
                            
                            # both 3D world handle bb points
                            handle_bb_front_0=[max(handle_bb[0][0], handle_bb[1][0]), handle_bb[0][1], handle_bb[0][2]]
                            handle_bb_front_1 = [max(handle_bb[0][0], handle_bb[1][0]), handle_bb[1][1], handle_bb[1][2]]

                            img_handle=[world_to_img(world_coord=handle_bb_front_0, projectionMatrix=projectionMatrix,
                                                        viewMatrix=viewMatrix, imwidth=width, imheight=height),
                                           world_to_img(world_coord=handle_bb_front_1, projectionMatrix=projectionMatrix,
                                                        viewMatrix=viewMatrix, imwidth=width, imheight=height)]

                            img_handle_large=get_min_max_bb(bb_in=img_handle, buffer_hor=int(width/40), buffer_ver=int(height/40))

                            plane_bb_im = [world_to_img(world_coord=plane_bb[0], projectionMatrix=projectionMatrix,
                                                        viewMatrix=viewMatrix, imwidth=width, imheight=height),
                                           world_to_img(world_coord=plane_bb[1], projectionMatrix=projectionMatrix,
                                                        viewMatrix=viewMatrix, imwidth=width, imheight=height)]

                            door_name=os.path.basename(urdf_input.replace('.urdf', ''))
                            env_name=os.path.basename(env_input.replace('.urdf',''))
                            
                            #Name of the img/json files
                            sample_name=str(env_name+"_"+door_name+"_ex_" + str('%.4s'%eye_x) + "_ey_" + str('%.4s'%eye_y) + "_ez_" + str(
                                    '%.4s'%eye_z) + "_ty_" + str('%.4s'%tar_y) + "_tz_" + str('%.4s'%tar_z)+ "_da_" + str('%.4s'%door_angle)+".FORMAT")



                            axis_is_right, axis = get_rotation_axis(plane_bb, handle_bb)
                            axis_img = [
                                world_to_img(world_coord=axis[0], projectionMatrix=projectionMatrix, viewMatrix=viewMatrix,
                                             imwidth=width, imheight=height),
                                world_to_img(world_coord=axis[1], projectionMatrix=projectionMatrix, viewMatrix=viewMatrix,
                                             imwidth=width, imheight
                                             
                                             
                                             =height)]


                            # Create datapoint if image fulfills quality criteria
                            if good_image(plane_bb_img=plane_bb_im, handle_bb_img=img_handle, axis_img=axis_img, imwidth=width, imheight=height, min_ec=0.5, ec_weight=1.0,
                                          iou_weight=1.0, min_score=0.7):
                                generate_datapoint(sample_name,
                                                   bb_door=get_min_max_bb(bb_in=plane_bb_im),
                                                   bb_handle=get_min_max_bb(bb_in=[clip_axis_point(img_handle_large[0],img_handle_large[1],400,400),clip_axis_point(img_handle_large[1],img_handle_large[0],400,400)]),
                                                   rotation=[clip_axis_point(axis_img[0],axis_img[1],400,400),clip_axis_point(axis_img[1],axis_img[0],400,400)], axis_is_right=axis_is_right, json_path=None,
                                                   rgb_img=rgbImg, depth_img=depthImg,
                                                   seg_img=segImg)
                                
# Returning 2D bb in the following format [[min_x, min_y], [max_x, max_y]]
def get_min_max_bb(bb_in, buffer_hor=0, buffer_ver=0):
    max_x = max(bb_in[0][0], bb_in[1][0]) + buffer_hor
    min_x = min(bb_in[0][0], bb_in[1][0]) - buffer_hor
    max_y = max(bb_in[0][1], bb_in[1][1]) + buffer_ver
    min_y = min(bb_in[0][1], bb_in[1][1]) - buffer_ver
    return [[min_x, min_y], [max_x, max_y]]

# Returns point clipped to img dims. target_point determines, 
# where the axis is heading to determine in which direction it points
def clip_axis_point(point, target_point, imwidth, imheight):
    if point[0]>=0 and point[0]<imwidth and point[1]>=0 and point[1]<imheight:
        return point
    else:
        left_axis=LineString([[0,0],[0, imheight]])
        right_axis=LineString([[imwidth,0],[imwidth,imheight]])
        bottom_axis=LineString([[0,imheight],[imwidth,imheight]])
        top_axis=LineString([[0,0],[imwidth,0]])

        point_line=LineString([point, target_point])

        if point_line.crosses(left_axis):
            point=[point_line.intersection(left_axis).x,point_line.intersection(left_axis).y]
        elif point_line.crosses(right_axis):
            point=[point_line.intersection(right_axis).x,point_line.intersection(right_axis).y]
        elif point_line.crosses(top_axis):
            point=[point_line.intersection(top_axis).x,point_line.intersection(top_axis).y]
        elif point_line.crosses(bottom_axis):
            point=[point_line.intersection(bottom_axis).x,point_line.intersection(bottom_axis).y]
        return [int(np.clip(point[0], 0, imwidth-1)), int(np.clip(point[1], 0, imheight-1))]



def main(*argv):
    #p.connect(p.GUI)
    p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(1. / 240.)

    floor = os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
    #p.loadMJCF(floor)
    env = p.loadURDF(argv[1])
    obj = p.loadURDF(argv[0])
    p.resetBasePositionAndOrientation(obj, [0.01,0,0.0], p.getBasePositionAndOrientation(obj)[1])


    # Plane Bounding Box
    plane_bb = p.getAABB(obj, linkIndex=0)
    # drawAABB(plane_bb)

    # Handle Bounding Box
    handle_bb = p.getAABB(obj, linkIndex=1)
    # drawAABB(handle_bb)

    # Get rotation axis
    #axis_is_left, axis=get_rotation_axis(plane_bb=plane_bb, handle_bb=handle_bb)
    #drawAABB(axis)
    
    # Sets target positions/orientations of camera
    eye_xs = np.linspace(3, 1, 3)
    eye_ys = np.linspace(-1.0, 1.0, 3)
    eye_zs = np.linspace(0.5, 2.0, 3)

    tar_ys = np.linspace(-1.0, 1.0, 4)
    tar_zs = np.linspace(0.0, 2, 3)

    generate_data_imgs(obj=obj, urdf_input=argv[0], env_input=argv[1],
                       eye_xs=eye_xs, eye_ys=eye_ys, eye_zs=eye_zs, tar_ys=tar_ys, tar_zs=tar_zs, door_angles=[0.0])

    sys.exit()

    for _ in range(24000):  # at least 100 seconds
        p.stepSimulation()
        time.sleep(1. / 240.)


    p.disconnect()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("No urdf specified. Aborting!")
        exit(-1)
    urdf_input = sys.argv[1]
    env_input = sys.argv[2]

    main(urdf_input, env_input)
