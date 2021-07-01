import os
import re
import json
from sys import settrace
import time
import argparse
import numpy as np
from natsort import natsorted as sorted
import glob

# XML-Stuff
import xacro
from xml.dom import minidom
import subprocess


# (X, Y, Z) = (Depth, Width, Height)
# door = (4, 75, 200), cabinet = (2, 45, 170), cupboard = (1.5, 50, 60)
standard_scaling = {
    'door':     np.array([1.0, 1.0, 1.0]), 
    'cabinet':  np.array([0.5, 0.6, 0.85]), 
    'cupboard': np.array([3.0/8.0, 2.0/3.0, 0.3])
}

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
        json_path = os.path.abspath(find_files('door.json', '../')[0])
    with open(json_path,) as file:
        data = json.load(file)
    return data

def get_scale(hx):
    doc = read_xml(hx)
    meshs = doc.getElementsByTagName('mesh') # get mesh data from joint
    mesh = meshs[0]
    if mesh.hasAttribute('scale'):
        scales_str = mesh.attributes['scale'].value
        sx, sy, sz = scales_str.split(' ') # ensures the ordering is correct
        sx = float(sx)
        sy = float(sy)
        sz = float(sz)
    return [sx, sy, sz]

def rescale_handle(doc, plane_xacro):
    scale = get_scale(plane_xacro)
    sx, sy, sz = scale
    # alter the final xacro (rescale the handle size by the sz-factor)
    links = doc.getElementsByTagName('link')[-1] # last link is the handle
    mesh_nodes = links.getElementsByTagName('mesh')

    for node in mesh_nodes:
        scale = ' '.join([str(float(x) * sy) for x in node.getAttribute('scale').split(' ')])
        node.attributes['scale'].value = scale
 
    return doc

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

def get_handle_config(scale=1.0): # use y-scale from door-xacro
    pos_y = np.random.choice([-0.3, 0.3]) * scale
    r = np.pi if pos_y > 0 else 0 # orientation
    return [0, pos_y, 0], [r, 0, 0]

def process_xacro(xacro_in, xacro_out, kwargs):
    in_file =  open(xacro_in,  'r')    # open the template xacro 
    out_file = open(xacro_out, 'w')  # write the xacro with parameters filled in

    # find and replace
    for in_line in in_file:
        out_line = in_line.replace("${plane_xacro}",  str(kwargs['plane_xacro']))
        out_line = out_line.replace("${handle_xacro}", str(kwargs['handle_xacro']))
        out_line = out_line.replace("${handle_pos_x}", str(kwargs['handle_pos_x']))
        out_line = out_line.replace("${handle_pos_y}", str(kwargs['handle_pos_y']))
        out_line = out_line.replace("${handle_pos_z}", str(kwargs['handle_pos_z']))
        out_line = out_line.replace("${handle_ori_r}", str(kwargs['handle_ori_r']))
        out_line = out_line.replace("${handle_ori_p}", str(kwargs['handle_ori_p']))
        out_line = out_line.replace("${handle_ori_y}", str(kwargs['handle_ori_y']))
        # print(out_line, end='')
        out_file.write(out_line)
    out_file.close()

def generate_door_xacro(**kwargs):
    out_path = '../data/objs/generated_objs/generated_doors/'
    xacro_path = out_path + 'complete_door.xacro'

    door_path = kwargs['plane_xacro']
    door_identifier = '.'.join(door_path.split('/')[-1].split('.')[:-1]) # name of door without file_ending
    handle_path = kwargs['handle_xacro']
    handle_identifier = re.search('[0-9]+_[0-9]+', handle_path.split('/')[-1]).group(0) # number of handle
    xacro_output = out_path + 'xacro/' + '{}_h_{}'.format(door_identifier, handle_identifier) + '.xacro'

    process_xacro(xacro_path, xacro_output, kwargs)
    print('xacro_output: ', xacro_output)

    ## temporatily copy xacro-file to be processed to scripts directory ##
    os.system("cp " + xacro_output + ' .')
    xacro_file = xacro_output.split('/')[-1]
    urdf_file = xacro_file.replace('.xacro', '.urdf')

    ## Process Xacro-File ##
    args = [xacro_file]
    opts, input_file_name = xacro.process_args(args)
    doc = xacro.process_file(input_file_name, **vars(opts))

    # DEPRECATED: check if cas (cabinet) or cus (supboard is specified)
    # -> also relscaled "normal" doors require rescaling of handles
    # door_type = len(re.findall('c[a|u]s', door_identifier))
    # if door_type > 0:
    plane_xacro = str(kwargs['plane_xacro'])
    doc = rescale_handle(doc, plane_xacro)

    ## Generate urdf-file in the designated folder ##
    urdf_out = out_path + 'urdf/' + urdf_file
    save_xml(urdf_out, doc)
    os.system('rm ' + os.getcwd() + '/' + xacro_file) # cleaning up

def generate_doors(data, door_path, handle_path):
    print(door_path, '\n', handle_path)
    door_types = list(data['doors'].keys())
    print(door_types)

    count = 0

    for door_type in door_types:
        print('Processing [{}]'.format(door_type))

        # get objects and textures of current door
        door_objects = data['doors'][door_type]['objects']        
        if len(door_objects) == 0:
            continue

        tex = data['doors'][door_type]['texture'] # get available textures of current door type
        print('doors = ', door_objects); print('texture = ', tex)

        for door in door_objects:    # possible objects
            print('obj = ', door)
            for dt in tex:   # possible textures

                door_filename = 'door_{}_{}_{}'.format(door, dt, door_type[:2])
                path = os.path.join(door_path, door_filename)

                # get the current door-type with all available sizes
                door_files = glob.glob(path + '*.xacro')

                # now get the available handles to use with this door
                handle_types = data['doors'][door_type]['handle']

                for handle_type in handle_types:

                    handle_obj = data['handles'][handle_type]['objects']
                    handle_tex = data['handles'][handle_type]['texture']

                    for handle in handle_obj:

                        for ht in handle_tex:
                            
                            handle_filename = 'handle_{}_{}.xacro'.format(handle, ht)

                            hx = os.path.join(handle_path, handle_filename)
                            for dx in door_files:
                                scale = get_scale(dx)
                                xyz, rpy = get_handle_config(scale[1])
                                generate_door_xacro(plane_xacro=dx, handle_xacro=hx,
                                    handle_pos_x=xyz[0], handle_pos_y=xyz[1], handle_pos_z=xyz[2],
                                    handle_ori_r=rpy[0], handle_ori_p=rpy[1], handle_ori_y=rpy[2])
                                count += 1
                                print('xacro[{}]'.format(count), end='\t')

def main(input_args):

    if input_args.type[0] == 'door':
        door_path = '../data/objs/pieces/doors/xacro/'
        handle_path = '../data/objs/pieces/handles/xacro/'
        data = get_json_data()
        generate_doors(data, door_path, handle_path)
    else:
        print('No object type specified. Aborting!')
        exit(-1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generating combined xacros')
    parser.add_argument('-type', type=str, nargs='+', help='Type of xacro to create')
    input_args = parser.parse_args()
    main(input_args)