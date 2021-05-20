import os
from sys import settrace
import time
import argparse
import numpy as np
from natsort import natsorted as sorted

# XML-Stuff
import xacro
from xml.dom import minidom
import subprocess

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


def generate_xacro(xacro_path, obj_type, **kwargs):
    # door: size_xyz, mesh
    # handle: mesh

    xacro_output = kwargs[str(obj_type) + '_mesh_file'].split('/')[-1].replace('.obj', '_s{}.xacro'.format(kwargs['size_x']))
    out_path = "/".join(xacro_path.split("/")[:-1]) + '/xacro/' + xacro_output

    in_file = open(xacro_path, 'r')
    out_file = open(out_path, 'w')

    for in_line in in_file:
        if obj_type == 'door':
            in_line = in_line.replace("${z_origin}", kwargs['size_z'])
            in_line = in_line.replace("${size_x}", kwargs['size_x'])
            in_line = in_line.replace("${size_y}", kwargs['size_y'])
            in_line = in_line.replace("${size_z}", kwargs['size_z'])
        mesh = "${"+ str(obj_type) + "_mesh_file}"
        in_line = in_line.replace(mesh, kwargs[str(obj_type) + '_mesh_file'])
        # print(in_line, end='')
        out_file.write(in_line)

    # print("Processed Xacro saved to: ", os.path.relpath(out_path))


def main(input_args):
    
    obj_type = input_args.type[0]  # type of objects

    # check in input_args, which directory to choose
    if obj_type == "door":
        xacro_path = os.path.abspath('../data/objs/pieces/' + obj_type + 's/plane.xacro')
    elif obj_type == "handle":
        xacro_path = os.path.abspath('../data/objs/pieces/' + obj_type + 's/handle.xacro')
    else:
        print('Invalid Object type specified. Aborting!')
        exit(-1)

    # load xacro file and generate new xacros
    print('Found xacro template for a {} in: {}'.format(obj_type, xacro_path))

    # path to the objects
    mesh_path = '/'.join(xacro_path.split('/')[:-1]) + '/mesh/'
    avail_files = sorted([mesh_path + obj for obj in os.listdir(mesh_path) if (mesh_path + obj).endswith('obj')])

    avail_obj = []
    avail_tex = []

    # exctract all the combinations of object, texture that are available (assumption: every object has all textures applied)
    for obj in avail_files: 
        file = obj.split('/')[-1].split('.')[0] # get filename without ending
        _, door_nr, tex_nr = file.split('_')
        avail_obj.append(door_nr)
        avail_tex.append(tex_nr)

    avail_obj = sorted(np.unique(avail_obj))
    avail_tex = sorted(np.unique(avail_tex))

    scales = [str(round(s, 2)) for s in np.linspace(1.0, 2.0, 11)] # strings for scaling of the door

    for obj_nr in avail_obj: # for loop over the different available objects
        for tex_nr in avail_tex:  # for-loop over the different available objects

            obj_file = '{}_{}_{}.obj'.format(obj_type, obj_nr, tex_nr)
            mesh = mesh_path + obj_file
            mesh = os.path.relpath(mesh)

            for s in scales:

                if obj_type == 'door':
                    generate_xacro(xacro_path, obj_type, door_mesh_file=mesh,
                                   size_x=s, size_y=s, size_z=s)    # size could be given as an input parameter
                if obj_type == 'handle':
                    generate_xacro(xacro_path, obj_type, handle_mesh_file=mesh)
    print('Processing of {} {}-xacros -> Done!'.format(int(len(avail_obj) * len(avail_tex) * len(scales)), obj_type))
    print('#Doors: ', len(avail_obj))
    print('#Textures: ', len(avail_tex))
    print('#Scales: ', len(scales))
    return 

    # load all objects
    for obj_path in avail_obj:
        # get path to obj-files to use as mesh-parameter for the xacro
        mesh = os.path.relpath(obj_path)
        print(mesh)
        continue
        # for tex in textures: # TODO 2nd for-loop
        # for scale in scales: # TODO 3rd for-loop
        scales = [str(round(s, 2)) for s in np.linspace(1.0, 2.0, 11)] # strings for scaling of the door    
        for s in scales:
            if obj_type == 'door':
                # generate_xacro(xacro_path, obj_type, door_mesh_file=mesh,
                                # size_x='1.0', size_y='1.0', size_z='1.0')    # size could be given as an input parameter
                generate_xacro(xacro_path, obj_type, door_mesh_file=mesh,
                               size_x=s, size_y=s, size_z=s)    # size could be given as an input parameter

            if obj_type == 'handle':
                generate_xacro(xacro_path, obj_type, handle_mesh_file=mesh)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Objs into Xacro files (with given parameters)')
    parser.add_argument('-type', type=str, nargs='+', help='Type of xacro file to use (door or handle)')
    input_args = parser.parse_args()
    main(input_args)