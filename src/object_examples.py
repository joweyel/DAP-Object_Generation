from gibson2.objects.ycb_object import YCBObject
from gibson2.objects.articulated_object import ArticulatedObject
import gibson2
import os
import pybullet as p
import pybullet_data
import time
# import openmesh as om



def main():
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(1./240.)

    floor = os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
    p.loadMJCF(floor)

    urdf_path = 'urdf_xacro/'

    # door = os.path.join(gibson2.assets_path, 'models/cabinet/door.urdf')
    # door = 'door.urdf'
    # handle = 'door_handle.urdf'
    handle = urdf_path + 'out.urdf'

    # obj_door = ArticulatedObject(filename=door)
    # obj_door.load()
    # obj_door.set_position([0, 0, 0])

    obj_handle = ArticulatedObject(filename=handle)
    obj_handle.load()
    obj_handle.set_position([0, 0, 0])



    for _ in range(24000):  # at least 100 seconds
        p.stepSimulation()
        time.sleep(1./240.)

    p.disconnect()


if __name__ == '__main__':
    main()
