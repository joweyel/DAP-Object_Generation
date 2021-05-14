import os
import re
from sys import settrace
import time
import argparse
import numpy as np

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


def generate_door_xacro(n_door, n_handle, **kwargs):
    # Most parameters are fixed
    # number of door/handle can be chosen + parameters stil to be set
    out_path = '../data/objs/generated_objs/generated_doors/'
    xacro_path = out_path + 'complete_door.xacro'
    door_path = '../data/objs/pieces/doors/xacro/door' + str(n_door) + '.xacro'
    handle_path = '../data/objs/pieces/handles/xacro/handle' + str(n_handle) + '.xacro'

    print('out_path: ', out_path)
    print('xacro_path: ', xacro_path)
    print('door_path: ', door_path)
    print('handle_path: ', handle_path)
    xacro_output = out_path + 'xacro/' + 'door_{}_{}'.format(n_door, n_handle) + '.xacro'
    print('xacro_output: ', xacro_output)
    print(kwargs, '\n')

    in_file = open(xacro_path, 'r')
    out_file = open(xacro_output, 'w')

    for in_line in in_file:
        # meshes 
        in_line = in_line.replace("${plane_xacro}",  str(kwargs['plane_xacro']))
        in_line = in_line.replace("${handle_xacro}", str(kwargs['handle_xacro']))
        in_line = in_line.replace("${handle_pos_x}", str(kwargs['handle_pos_x']))
        in_line = in_line.replace("${handle_pos_y}", str(kwargs['handle_pos_y']))
        in_line = in_line.replace("${handle_pos_z}", str(kwargs['handle_pos_z']))
        in_line = in_line.replace("${handle_ori_r}", str(kwargs['handle_ori_r']))
        in_line = in_line.replace("${handle_ori_p}", str(kwargs['handle_ori_p']))
        in_line = in_line.replace("${handle_ori_y}", str(kwargs['handle_ori_y']))
        print(in_line, end='')
        out_file.write(in_line)

    '''
    plane_xacro=dx, handle_xacro=hx,
    handle_pos_x=0.0, handle_pos_y=0.3, handle_pos_z=0.0, 
    handle_ori_r=0.0, handle_ori_p=0.0, handle_ori_y=0.0) 
    '''

    # args = [xacro_path]
    # for key, value in kwargs.items():
    #     args.append(key + ":=" + str(value))
    # opts, input_file_name = xacro.process_args(args)
    # print(opts)
    # doc = xacro.process_file(input_file_name, **vars(opts))
    # return
    # xacro_output = out_path + 'xacro/' + 'door_{}_{}'.format(n_door, n_handle) + '.xacro'
    # TODO: 


def get_handle_config():
    pos_y = np.random.choice([-0.3, 0.3])
    r = 0 if pos_y > 0 else np.pi # orientation
    return [0, pos_y, 0], [r, 0, 0]

def main(input_args):
    n_door = str(input_args.door[0])
    n_handle = str(input_args.handle[0])
    # generate_door_xacro(n_door, n_handle)

    doors_path = '../data/objs/pieces/doors/xacro/'
    handles_path = '../data/objs/pieces/handles/xacro/'
    # get numbers of available doors/handles
    door_xacros = sorted([door.split('/')[-1] for door in os.listdir(doors_path)])
    door_numbers = [re.findall('[0-9]+', x)[0] for x in door_xacros] # get the numbers
    print(door_numbers, door_xacros)

    handle_xacros = sorted([handle.split('/')[-1] for handle in os.listdir(handles_path)])
    handle_numbers = [re.findall('[0-9]+', x)[0] for x in handle_xacros] # get the numbers
    # print(handle_xacros)
    # print(handle_numbers)

    for n_door in door_numbers:
        for n_handle in handle_numbers: # handle_numbers:
            dx = doors_path + 'door' + str(n_door) + '.xacro'
            hx = handles_path + 'handle' + str(n_handle) + '.xacro'

            xyz, rpy = get_handle_config()
            generate_door_xacro(n_door, n_handle, plane_xacro=dx, handle_xacro=hx,
                handle_pos_x=xyz[0], handle_pos_y=xyz[1], handle_pos_z=xyz[2], 
                handle_ori_r=rpy[0], handle_ori_p=rpy[1], handle_ori_y=rpy[2]) 
            
    return # for later

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate "whole door" xacro')
    parser.add_argument('-door', type=int, nargs='+', help='Path to Door plane')
    parser.add_argument('-handle', type=int, nargs='+', help='Path to Door handle')
    input_args = parser.parse_args()
    main(input_args)