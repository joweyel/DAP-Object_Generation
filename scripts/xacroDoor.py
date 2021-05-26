import os
import re
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


def generate_door_xacro(**kwargs):
    out_path = '../data/objs/generated_objs/generated_doors/'
    xacro_path = out_path + 'complete_door.xacro'

    door_path = kwargs['plane_xacro']
    door_identifier = '.'.join(door_path.split('/')[-1].split('.')[:-1]) # name of door without file_ending

    handle_path = kwargs['handle_xacro']
    handle_identifier = re.search('[0-9]+_[0-9]+', handle_path.split('/')[-1]).group(0) # number of handle
    xacro_output = out_path + 'xacro/' + '{}_h{}'.format(door_identifier, handle_identifier) + '.xacro'
    print(xacro_output)

    in_file = open(xacro_path, 'r')     # open the template xacro 
    out_file = open(xacro_output, 'w')  # write the xacro with parameters filled in

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

    print('xacro_output: ', xacro_output)

    ## temporatily copy xacro-file to be processed to scripts directory ##
    os.system("cp " + xacro_output + ' .')
    xacro_file = xacro_output.split('/')[-1]
    urdf_file = xacro_file.replace('.xacro', '.urdf')

    ## Process Xacro-File ##
    args = [xacro_file]
    opts, input_file_name = xacro.process_args(args)
    doc = xacro.process_file(input_file_name, **vars(opts))

    ## Generate urdf-file in the designated folder ##
    urdf_out = out_path + 'urdf/' + urdf_file
    save_xml(urdf_out, doc)
    os.system('rm ' + os.getcwd() + '/' + xacro_file) # cleaning up



def get_handle_config(scale=1.0):
    pos_y = np.random.choice([-0.3, 0.3]) * scale
    r = np.pi if pos_y > 0 else 0 # orientation
    return [0, pos_y, 0], [r, 0, 0]

def main():
    doors_path = '../data/objs/pieces/doors/xacro/'
    handle_path = '../data/objs/pieces/handles/xacro/'
    # get numbers of available doors/handles
    door_xacros = sorted([door.split('/')[-1] for door in os.listdir(doors_path)if door.split('/')[-1].endswith('xacro')])
    handle_xacros = sorted([handle.split('/')[-1] for handle in os.listdir(handle_path) if handle.split('/')[-1].endswith('xacro')])

    for d in door_xacros:
        for h in handle_xacros:
            dx = doors_path + d
            hx = handle_path + h            
            door_scale = float(re.search('[0-9]\.[0-9]', dx).group(0)) # extract the scale 
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