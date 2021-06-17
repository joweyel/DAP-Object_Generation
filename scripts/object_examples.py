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
from shapely.geometry import box, Polygon


def check_coverage(bb, imwidth, imheight, min_iou=0.9):
    im_boundary = np.copy(bb)
    im_boundary[0][0] = np.clip(im_boundary[0][0], 0, imwidth)
    im_boundary[1][0] = np.clip(im_boundary[1][0], 0, imwidth)
    im_boundary[0][1] = np.clip(im_boundary[0][1], 0, imheight)
    im_boundary[1][1] = np.clip(im_boundary[1][1], 0, imheight)

    iou = iou_coverage(bb, im_boundary)
    if iou > min_iou:
        return True
    else:
        return False


def iou_coverage(bb, image_bb):
    bb_edgepoints = bb_to_edgepoints(bb)
    im_edgepoints = bb_to_edgepoints(image_bb)
    bb_shape = Polygon(bb_edgepoints)
    im_shape = Polygon(im_edgepoints)

    intersection = bb_shape.intersection(im_shape).area
    union = bb_shape.union(im_shape).area
    iou = intersection / union
    return iou


def bb_to_edgepoints(bb):
    bb_min_x = min(bb[0][0], bb[1][0])
    bb_max_x = max(bb[0][0], bb[1][0])
    bb_min_y = min(bb[0][1], bb[1][1])
    bb_max_y = max(bb[0][1], bb[1][1])
    return [[bb_min_x, bb_min_y], [bb_min_x, bb_max_y], [bb_max_x, bb_max_y], [bb_max_x, bb_min_y]]


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

    return [x_scaled[0],y_scaled[1]]


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


def generate_datapoint(file_name, bb_door, bb_handle, rotation, json_path=None, **kwargs):
    '''
        Function: saves images and its corresponding features (in a json-file)
        Input:
            bb_door:   min/max-Points of 2D door bounding-box
            bb_handle: min/max-Points of 2D handle bounding-box
    '''

    output_path = '../data/train_data/'

    print(kwargs)

    data = get_json_data(json_path)
    data['object']['min'] = bb_door[0]
    data['object']['max'] = bb_door[1]
    data['handle']['min'] = bb_handle[0]
    data['handle']['max'] = bb_handle[1]
    data['axis'] = rotation

    file_name = os.path.basename(file_name)

    # save images
    if 'rgb_img' in kwargs.keys():
        rgb_out = os.path.join(output_path, 'images/') + 'rgb_' + file_name.replace('.urdf', '.png')  # or jpg
        plt.imsave(rgb_out, kwargs['rgb_img'])

    if 'depth_img' in kwargs.keys():
        depth_out = os.path.join(output_path, 'images/') + 'depth_' + file_name.replace('.urdf', '.png')
        plt.imsave(depth_out, kwargs['depth_img'])
    if 'seg_img' in kwargs.keys():
        seg_out = os.path.join(output_path, 'images/') + 'seg_' + file_name.replace('.urdf', '.png')
        plt.imsave(seg_out, kwargs['seg_img'])

    # save features in json
    json_out = os.path.join(output_path, 'features/') + file_name.replace('.urdf', '.json')
    with open(json_out, 'w') as f:
        json.dump(data, f, indent=4)


def main(urdf_input):
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(1. / 240.)

    floor = os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
    p.loadMJCF(floor)

    obj = p.loadURDF(urdf_input)

    # Plane Bounding Box
    plane_bb = p.getAABB(obj, linkIndex=0)
    # drawAABB(plane_bb)

    # Handle Bounding Box
    handle_bb = p.getAABB(obj, linkIndex=1)
    # drawAABB(handle_bb)

    # Complete Door Bounding Box
    all_bb_points = [plane_bb[0], plane_bb[1], handle_bb[0], handle_bb[1]]
    complete_bb = [np.min(all_bb_points, axis=0), np.max(all_bb_points, axis=0)]
    # drawAABB(complete_bb)

    # Get rotation axis
    handle_center_y = (handle_bb[0][1] + handle_bb[1][1]) / 2
    plane_center_y = (plane_bb[0][1] + plane_bb[1][1]) / 2
    if handle_center_y > plane_center_y:
        axis = [[(plane_bb[0][0] + plane_bb[1][0]) / 2, plane_bb[0][1], plane_bb[0][2]],
                [(plane_bb[0][0] + plane_bb[1][0]) / 2, plane_bb[0][1], plane_bb[1][2]]]
    else:
        axis = [[(plane_bb[0][0] + plane_bb[1][0]) / 2, plane_bb[1][1], plane_bb[0][2]],
                [(plane_bb[0][0] + plane_bb[1][0]) / 2, plane_bb[1][1], plane_bb[1][2]]]
    drawAABB(axis)

    eye_xs = np.linspace(3, 1, 4)
    eye_ys = np.linspace(-2, 2, 4)
    eye_zs = np.linspace(-0.5, 2.5, 4)

    tar_ys = np.linspace(-1.5, 1.5, 4)
    tar_zs = np.linspace(-0.5, 2, 4)

    for eye_x in eye_xs:
        for eye_y in eye_ys:
            for eye_z in eye_zs:
                for tar_y in tar_ys:
                    for tar_z in tar_zs:
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
                        plane_bb_im=[world_to_img(world_coord=plane_bb[0],projectionMatrix=projectionMatrix, viewMatrix=viewMatrix, imwidth=width, imheight=height),
                                     world_to_img(world_coord=plane_bb[1], projectionMatrix=projectionMatrix,viewMatrix=viewMatrix, imwidth=width, imheight=height)]

                        imcoord = world_to_img(world_coord=plane_bb[1], projectionMatrix=projectionMatrix,
                                               viewMatrix=viewMatrix, imwidth=width, imheight=height)
                        marked_rgbImg = cv2.circle(rgbImg, (int(imcoord[0]), int(imcoord[1])), radius=1,
                                                   color=(0, 0, 255), thickness=10)
                        if check_coverage(plane_bb_im, width, height):
                            cv2.imwrite("../data/train_data/imgs/pos_x"+str(eye_x)+"y"+str(eye_y)+"z"+str(eye_z)+"ty"+str(tar_y)+"tz"+str(tar_z)+".png", marked_rgbImg)
                        else:
                            cv2.imwrite("../data/train_data/imgs/neg_x"+str(eye_x)+"y"+str(eye_y)+"z"+str(eye_z)+"ty"+str(tar_y)+"tz"+str(tar_z)+".png", marked_rgbImg)

                        """generate_datapoint(urdf_input,
                                           bb_door=[[0, 0], [1, 1]],
                                           bb_handle=[[0.5, 0.5], [1, 1]],
                                           rotation=[[0.0, 1.0]], json_path=None,
                                           rgb_img=rgbImg, depth_img=depthImg,
                                           seg_img=segImg)
                        continue"""

    for _ in range(24000):  # at least 100 seconds
        p.stepSimulation()
        time.sleep(1. / 240.)

    p.disconnect()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("No urdf specified. Aborting!")
        exit(-1)
    urdf_input = sys.argv[1]

    main(urdf_input)
