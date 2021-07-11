import os
import re
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
        print(out_line, end='')
        # out_file.write(out_line)
    out_file.close()


def generate_door_xacro(**kwargs):
    out_path = '../data/objs/generated_objs/generated_doors/'
    xacro_path = out_path + 'complete_door.xacro'

    door_path = kwargs['plane_xacro']
    door_identifier = '.'.join(door_path.split('/')[-1].split('.')[:-1]) # name of door without file_ending
    print(door_identifier)
    handle_path = kwargs['handle_xacro']
    handle_identifier = re.search('[0-9]+_[0-9]+', handle_path.split('/')[-1]).group(0) # number of handle

    xacro_output = out_path + 'xacro/' + '{}_h{}'.format(door_identifier, handle_identifier) + '.xacro'
    print(xacro_output)

    # process_xacro(xacro_path, xacro_output, kwargs)
    
    return


    print('xacro_output: ', xacro_output)

    ## temporatily copy xacro-file to be processed to scripts directory ##
    os.system("cp " + xacro_output + ' .')
    xacro_file = xacro_output.split('/')[-1]
    urdf_file = xacro_file.replace('.xacro', '.urdf')

    ## Process Xacro-File ##
    args = [xacro_file]
    opts, input_file_name = xacro.process_args(args)
    doc = xacro.process_file(input_file_name, **vars(opts))
    print(doc)

    # check if cas (cabinet) or cus (supboard is specified)
    door_type = len(re.findall('c[a|u]s', door_identifier))
    if door_type > 0:
        plane_xacro = str(kwargs['plane_xacro'])
        doc = rescale_handle(doc, plane_xacro)

    ## Generate urdf-file in the designated folder ##
    urdf_out = out_path + 'urdf/' + urdf_file
    save_xml(urdf_out, doc)
    os.system('rm ' + os.getcwd() + '/' + xacro_file) # cleaning up



def get_handle_config(scale=1.0): # use y-scale from door-xacro
    pos_y = np.random.choice([-0.3, 0.3]) * scale
    r = np.pi if pos_y > 0 else 0 # orientation
    return [0, pos_y, 0], [r, 0, 0]

def main():
    doors_path = '../data/objs/pieces/doors/xacro/'
    handle_path = '../data/objs/pieces/handles/xacro/'

    # load from json:
    # -> load the json by the numbers of the doors and handles
    data = get_json_data()
    print(data)
    
    # get numbers of available doors/handles
    door_xacros = sorted([door.split('/')[-1] for door in os.listdir(doors_path)if door.split('/')[-1].endswith('xacro')])
    handle_xacros = sorted([handle.split('/')[-1] for handle in os.listdir(handle_path) if handle.split('/')[-1].endswith('xacro')])
    print(handle_xacros)
    return

    ## TODO:
    # when loading the files -> read the type of door and choose the possible 
    # objects/textures from the saved json-file to generate

    for d in door_xacros:
        for h in handle_xacros:
            print(d, h)
            continue
            dx = doors_path + d
            hx = handle_path + h
            # get the y-scale to correctly scale the position of handles
            scale = get_scale(dx)
            xyz, rpy = get_handle_config(scale[1])
            generate_door_xacro(plane_xacro=dx, handle_xacro=hx,
                handle_pos_x=xyz[0], handle_pos_y=xyz[1], handle_pos_z=xyz[2],
                handle_ori_r=rpy[0], handle_ori_p=rpy[1], handle_ori_y=rpy[2])


    return

    for d in door_xacros:
        for h in handle_xacros:
            dx = doors_path + d
            hx = handle_path + h            
            door_scale = float(re.search('[0-9]\.[0-9]', dx).group(0)) # extract the scale 
            # get y-scale from xacro
            xyz, rpy = get_handle_config(door_scale)
            generate_door_xacro(plane_xacro=dx, handle_xacro=hx,
                handle_pos_x=xyz[0], handle_pos_y=xyz[1], handle_pos_z=xyz[2],
                handle_ori_r=rpy[0], handle_ori_p=rpy[1], handle_ori_y=rpy[2])

    return
    handle_xacros = sorted([handle.split('/')[-1] for handle in os.listdir(handles_path)])
    handle_numbers = [re.findall('[0-9]+', x)[0] for x in handle_xacros] # get the numbers
    print(handle_xacros)
    print(handle_numbers)
    
    return
    if n_door in door_numbers and n_handle in handle_numbers:
        dx = doors_path + 'door' + str(n_door) + '.xacro'
        hx = handles_path + 'handle' + str(n_handle) + '.xacro'

        xyz, rpy = get_handle_config()
        generate_door_xacro(n_door, n_handle, plane_xacro=dx, handle_xacro=hx,
            handle_pos_x=xyz[0], handle_pos_y=xyz[1], handle_pos_z=xyz[2], 
            handle_ori_r=rpy[0], handle_ori_p=rpy[1], handle_ori_y=rpy[2])
        
    

if __name__ == '__main__':
    main()