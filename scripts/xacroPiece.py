import os
import json
from sys import settrace
import time
import argparse
import numpy as np
from natsort import natsorted as sorted

# XML-Stuff
import xacro
from xml.dom import minidom
import subprocess

# (X, Y, Z) = (Depth, Width, Height) in Pybullet
# door = (4, 75, 200), cabinet = (2, 45, 170), cupboard = (1.5, 50, 60)
# dictionary with scaling factors for certain door types
standard_scaling = {
    'door': np.array([1.0, 1.0, 1.0]), 
    'cabinet': np.array([0.5, 0.6, 0.85]), 
    'cupboard': np.array([3.0/8.0, 2.0/3.0, 0.3])
}

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

def print_file(file):
    with open(file, 'r') as f:
        for line in f:
            print(line, end='')

def get_xacro_template(obj_type):
    '''
        Function:
            gets the path to the xacro-template for a certain object type
        Input:  type of object
        Output: path to xacro-template for a certain object type
    '''
    if obj_type == 'door':
        return os.path.abspath('../data/objs/pieces/' + obj_type + 's/plane.xacro')
    elif obj_type == "handle":
        return os.path.abspath('../data/objs/pieces/' + obj_type + 's/handle.xacro')
    else:
        print('Invalid Object type specified. Aborting!')
        exit(-1)

def find_files(filename, search_path):
   result = []
    # Wlaking top-down from the root
   for root, dir, files in os.walk(search_path):
      if filename in files:
         result.append(os.path.join(root, filename))
   return result

def generate_xacro(xacro_path, obj_type, cls, **kwargs):
    # door: size_xyz, mesh
    # handle: mesh)
    if obj_type == 'door': # output_name for the door(like)-output
        door_prefix = cls[:2] if cls == 'cabinet' or cls == 'cupboard' else obj_type[:2]
        if cls == None:
            scale = kwargs['size_x']
        else:
            scale = 1.0
        xacro_output = kwargs[str(obj_type) + '_mesh_file']
        xacro_output = xacro_output.split('/')[-1].replace('.obj', '_' + door_prefix + 's{}.xacro'.format(scale))

    if obj_type == 'handle': # output_name for the handle-object
        xacro_output = kwargs[str(obj_type) + '_mesh_file'].split('/')[-1].replace('.obj', '.xacro')

    out_path = os.path.join("/".join(xacro_path.split("/")[:-1]), 'xacro', xacro_output)
    
    in_file = open(xacro_path, 'r')
    out_file = open(out_path, 'w')

    path = os.path.relpath(kwargs[str(obj_type) + '_mesh_file'])
    # fill in the parameters of the specified xacro
    for in_line in in_file:
        if obj_type == 'door':
            in_line = in_line.replace("${z_origin}", kwargs['size_z'])
            in_line = in_line.replace("${size_x}", kwargs['size_x'])
            in_line = in_line.replace("${size_y}", kwargs['size_y'])
            in_line = in_line.replace("${size_z}", kwargs['size_z'])
        mesh = "${"+ str(obj_type) + "_mesh_file}"
        in_line = in_line.replace(mesh, path)
        # print(in_line, end='')
        out_file.write(in_line)

    print("Processed Xacro saved to: ", os.path.relpath(out_path))
    in_file.close()
    out_file.close()
    
    return

def generate_xacros(xacro_path, mesh_path, obj_type, cls, obj, tex):

    scales = [str(round(s, 2)) for s in np.linspace(0.5, 1.0, 11)] # strings for scaling of the door
    for obj_nr in obj:
        for tex_nr in tex:
            mesh = mesh_path + '{}_{}_{}.obj'.format(obj_type, obj_nr, tex_nr)
            mehs = os.path.relpath(mesh)
            if os.path.isfile(mesh) == False:
                continue    # skip texture if not available

            if obj_type == 'door':

                if cls == 'cabinet' or cls == 'cupboard': # both cases can be handled here just by
                    scaling_factor = standard_scaling[cls]
                    generate_xacro(xacro_path, obj_type, cls, door_mesh_file=mesh,
                                   size_x=str(scaling_factor[0]), 
                                   size_y=str(scaling_factor[1]), 
                                   size_z=str(scaling_factor[2]))
                else: # normal door
                    for s in scales:
                        generate_xacro(xacro_path, obj_type, cls, door_mesh_file=mesh,
                                       size_x=s, size_y=s, size_z=s)

            if obj_type == 'handle':
                generate_xacro(xacro_path, obj_type, cls, handle_mesh_file=mesh)

    s = len(scales) if obj_type == 'door' else 1
    print('Processing of {} {}-xacros -> Done!'.format(int(len(obj) * len(tex) * s), obj_type))
    print('#Doors: ', len(obj))
    print('#Textures: ', len(tex))
    print('#Scales: ', len(scales))
    return

def get_json_data(file_path=None):
    if file_path:
        json_path = os.path.abspath(file_path)
    else:
        json_path = os.path.abspath(find_files('door.json', '../')[0])
    with open(json_path,) as file:
        data = json.load(file)
    return data

def get_category_data(data, obj_type, cls):
    if obj_type == 'door':
        if cls is not None:
            return data[obj_type + 's'][cls]
        else:
            return data[obj_type + 's'][obj_type]
    elif obj_type == 'handle':
        return data[obj_type + 's']
    else:
        print('Object-Type not supported')
        exit(-1)


def main(input_args):
    obj_type = input_args.type[0]
    cls = input_args.cls
    print(obj_type, cls)

    # get the path of the xacro-template to use
    xacro_path = get_xacro_template(obj_type)
    print('Found xacro template for a {} in: {}'.format(obj_type, xacro_path))

    # load json to get informations about the xacros of objects to create
    data = get_json_data()

    # path to the objects
    mesh_path = '/'.join(xacro_path.split('/')[:-1]) + '/mesh/'
    ## TODO: HERE I NEED TO MAKE IT POSSIBLE TO GET ALL HANDLE TYPES 
    # if obj_type == 'door' and cls == None:
    #     category_data = data[obj_type + 's'][obj_type]
    # else 
    category_data = get_category_data(data, obj_type, cls)



    if obj_type == 'handle':
        for category in category_data:
            print('\nHandles of type [{}]'.format(category))
            avail_obj = category_data[category]['objects']
            avail_tex = category_data[category]['texture']
            generate_xacros(xacro_path, mesh_path, obj_type, cls, avail_obj, avail_tex)
    if obj_type == 'door':
        avail_obj = category_data['objects']
        avail_tex = category_data['texture']
        generate_xacros(xacro_path, mesh_path, obj_type, cls, avail_obj, avail_tex)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Objs into Xacro files (with given parameters)')
    parser.add_argument('-type', type=str, nargs='+', help='Type of xacro file to use (door or handle)')
    parser.add_argument('-cls', type=str, default=None, help='Class of door (usable when choosing door-type')
    input_args = parser.parse_args()
    main(input_args)