import os
import time

import rospkg
import rospy
import xacro
from xml.dom import minidom
import subprocess

from gazebo_msgs.srv import *


def save_xml(file, doc):
    try:
        out = open(file, 'w')
        out.write(doc.toprettyxml(indent='  '))
        out.close()
    except IOError as e:
        raise FileExistsError("Failed to open output:", exc=e)


def read_xml(file):
    try:
        doc = minidom.parse(file)
        return doc
    except IOError as e:
        raise FileExistsError("Failed to open output:", exc=e)


def convert_urdf_to_sdf(urdf_file):
    sdf_doc = subprocess.check_output(['gz', 'sdf', '-p', urdf_file])
    return sdf_doc


def remove_tag_velocity_decay(doc):
    nodes = doc.getElementsByTagName('velocity_decay')
    nodes[0].parentNode.removeChild(nodes[0])
    return doc


class Spawner(object):
    def __init__(self):
        self._delete_gazebo_model = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel, persistent=True)
        self._puck_spawner_service = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel, persistent=True)

    def spawn_puck(self, **kwargs):

        req = DeleteModelRequest()
        req.model_name = "puck_gazebo"
        self._delete_gazebo_model(req)
        time.sleep(0.1)

        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('air_hockey_description')
        puck_urdf_xacro = "urdf/puck/model.urdf.xacro"
        args = [os.path.join(pkg_path, puck_urdf_xacro)]

        for key, value in kwargs.items():
            args.append(key + ":=" + str(value))

        opts, input_file_name = xacro.process_args(args)
        doc = xacro.process_file(input_file_name, **vars(opts))

        urdf_file = os.path.join(pkg_path, "urdf/puck/model.urdf")
        save_xml(urdf_file, doc)

        sdf_file = os.path.join(pkg_path, "urdf/puck/model.sdf")
        sdf_doc = minidom.parseString(convert_urdf_to_sdf(urdf_file))
        sdf_doc = remove_tag_velocity_decay(sdf_doc)

        save_xml(sdf_file, sdf_doc)

        puck_srv = SpawnModelRequest()
        puck_srv.model_name = 'puck_gazebo'
        puck_srv.model_xml = sdf_doc.toxml()
        res = self._puck_spawner_service(puck_srv)
        return res.success

    def spawn_table(self, **kwargs):
        req = DeleteModelRequest()
        req.model_name = "air_hockey_table"
        res = self._delete_gazebo_model(req)
        time.sleep(0.1)

        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('air_hockey_description')
        puck_urdf = "urdf/air_hockey_table/model.urdf.xacro"
        puck_urdf_dir = os.path.join(pkg_path, puck_urdf)
        #
        args = [puck_urdf_dir]

        for key, value in kwargs.items():
            args.append(key + ":=" + str(value))

        opts, input_file_name = xacro.process_args(args)
        doc = xacro.process_file(input_file_name, **vars(opts))
        rospy.set_param('/air_hockey_table/robot_description', doc.toxml())

        puck_srv = SpawnModelRequest()
        puck_srv.model_name = 'air_hockey_table'
        puck_srv.model_xml = doc.toxml()
        res = self._puck_spawner_service(puck_srv)
        return res.success


# if __name__ == "__main__":
#     spawn_puck(puck_name='PuckGazebo', restitution=0.58, lateral_friction=0.111, spinning_friction=0.222,
#                linear_vel_decay=0.002, angular_vel_decay=0.002)
#     spawn_table(parent='Table', restitution_longitude=0.666, restitution_latitude=0.888, lateral_friction_rim=0.333, lateral_friction_surface=0.444)
