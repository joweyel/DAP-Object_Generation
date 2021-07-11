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


def save_xml(file, doc):
    try:
        out = open(file, 'w')
        out.write(doc.toprettyxml(indent='  '))
        out.close()
    except IOError as e:
        raise FileExistsError("Failed to open output:", exc=e)

def process_parameters(xacro_path, **kwargs):
    print(kwargs)
    in_file = open(xacro_path, 'r')
    out_file = open(kwargs['obj_file'].replace('.obj', '.xacro'), 'w')
    for in_line in in_file:
        in_line = in_line.replace('${floor_mesh_file}', str(kwargs['floor_mesh_file']))
        print(in_line, end='')
        out_file.write(in_line)
    pass

def generate_room(xacro_path):
    pass


def generate_floor(xacro_path):
    floor_mesh_path = '/'.join(xacro_path.split('/')[:-1]) + '/mesh/'
    print(floor_mesh_path)
    objs = sorted([obj for obj in os.listdir(floor_mesh_path) if obj.endswith('obj')])
    print(objs)

    # args = [os.path.join(os.getcwd(), door_path)] #

    # for key, value in kwargs.items():
    #     args.append(key + ":=" + str(value))
    # print(args)
    
    # opts, input_file_name = xacro.process_args(args)
    # print(opts, input_file_name)
    # doc = xacro.process_file(input_file_name, **vars(opts))


    for obj_file in objs:
        obj_path = floor_mesh_path + obj_file
        process_parameters(xacro_path, obj_file=obj_file, floor_mesh_file=obj_path)

        pass
        #args = [xacro_path, 'floor_mesh_file:=' + floor_mesh_path + obj_file]
        #opts, input_file_name = xacro.process_args(args)
        #print(opts, input_file_name)
        #doc = xacro.process_file(input_file_name, **vars(opts))
        #save_xml(obj_file + '.xacro', doc)
        #pretty_xml_as_string = doc.toprettyxml()
        #print(pretty_xml_as_string)

def main():
    floor_path = '../data/objs/pieces/floor/'
    floor_xacro_template = '../data/objs/pieces/floor/floor.xacro'
    generate_floor(floor_xacro_template)
    pass

if __name__ == '__main__':
    main()