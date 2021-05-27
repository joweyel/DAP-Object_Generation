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


from matplotlib.patches import Circle, Rectangle

def generate_bb_template(debug=False):
    size = (480, 640)
    img = np.zeros(size) 
    img[100, 100] = 255.
    img[350, 100] = 255.
    img[200, 300] = 255.
    img[450, 300] = 255.
    plt.imsave('bb_img.png', img)
    if debug:
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        circ1 = Circle((100, 100), 4, fill=False)
        circ2 = Circle((100, 350), 4, fill=False)
        circ3 = Circle((300, 200), 4, fill=False)
        circ4 = Circle((300, 450), 4, fill=False)

        ax.add_patch(circ1)
        ax.add_patch(circ2)
        ax.add_patch(circ3)
        ax.add_patch(circ4)
        plt.show()
    return img

from xml.dom import minidom
def getBB_data(img, debug=False):
    y, x = np.where(img == 255.)
    min_x = x.min()
    max_x = x.max()
    min_y = y.min()
    max_y = y.max()

    max_p = [max_y, max_x]
    min_p = [min_y, min_x]

    print('max_p = ', max_p)
    print('min_p = ', min_p)
    # draw bb in image
    if debug:
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        circ1 = Circle((min_x, min_y), 4, fill=False, edgecolor='g')
        circ2 = Circle((max_x, min_y), 4, fill=False)
        circ3 = Circle((min_x, max_y), 4, fill=False)
        circ4 = Circle((max_x, max_y), 4, fill=False, edgecolor='g')
        rect = Rectangle(min_p, max_x - min_x, max_y - min_y, 
            linewidth=1, edgecolor='r',facecolor='none')

        ax.add_patch(circ1)
        ax.add_patch(circ2)
        ax.add_patch(circ3)
        ax.add_patch(circ4)
        ax.add_patch(rect)
        ax.set_title("reconstructed")
        plt.show()
    
    root = minidom.Document()
    # generate root node
    xml = root.createElement('features')
    root.appendChild(xml)

    # create feature nodes
    obj = root.createElement('object')
    obj.setAttribute('max', '{} {}'.format(max_y, max_x))
    obj.setAttribute('min', '{} {}'.format(min_y, min_x))
    xml.appendChild(obj)


    handle = root.createElement('handle')
    handle.setAttribute('max', '{} {}'.format(max_y, max_x))
    handle.setAttribute('min', '{} {}'.format(min_y, min_x))
    xml.appendChild(handle)

    axis = root.createElement('axis')
    axis.setAttribute('p_low',  '{} {} {}'.format(0, -0.35, 0))
    axis.setAttribute('p_high', '{} {} {}'.format(0, -0.35, 2))
    xml.appendChild(axis)

    xml_str = root.toprettyxml(indent='\t')
    print(xml_str)

    save_path_file = "bb.xml"
    with open(save_path_file, "w") as f:
        f.write(xml_str) 

def get_features(xml_path):
    doc = minidom.parse(xml_path).documentElement
    obj = doc.getElementsByTagName('object')[0]
    obj_min_p = obj.attributes['min'].value
    obj_min_p = np.array([int(x) for x in obj_min_p.split(' ')])

    obj_max_p = obj.attributes['max'].value
    obj_max_p = np.array([int(x) for x in obj_max_p.split(' ')])
    
    print('obj_min_p = ', obj_min_p)
    print('obj_max_p = ', obj_max_p)

    handle = doc.getElementsByTagName('handle')[0]
    handle_min_p = handle.attributes['min'].value
    handle_min_p = np.array([int(x) for x in handle_min_p.split(' ')])

    handle_max_p = handle.attributes['max'].value
    handle_max_p = np.array([int(x) for x in handle_max_p.split(' ')])

    print('handle_min_p = ', handle_min_p)
    print('handle_max_p = ', handle_max_p)

    
    

def main(urdf_input):

    img = generate_bb_template(debug=True)
    getBB_data(img, debug=True)
    get_features('bb.xml')

    return

    p.connect(p.GUI)
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(1./240.)

    floor = os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
    p.loadMJCF(floor)

    obj = p.loadURDF(urdf_input)

    #Plane Bounding Box
    plane_bb=p.getAABB(obj, linkIndex=0)
    with open('bb.txt', 'w') as f:
        f.write('{}, {}'.format(plane_bb[0], plane_bb[1]))
    #drawAABB(plane_bb)

    #Handle Bounding Box
    handle_bb = p.getAABB(obj, linkIndex=1)
    #drawAABB(plane_bb)

    #Complete Door Bounding Box
    all_bb_points=[plane_bb[0], plane_bb[1], handle_bb[0], handle_bb[1]]
    complete_bb=[np.min(all_bb_points,axis=0),np.max(all_bb_points,axis=0)]
    drawAABB(complete_bb)




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

