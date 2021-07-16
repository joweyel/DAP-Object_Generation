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
        raise Exception("Failed to open output:", exc=e)


def process_xacro2(xacro_path, **kwargs):
    in_file = open(xacro_path, 'r')
    fmf=os.path.basename(kwargs['floor_mesh_file']).split('.')[0]
    wmf = os.path.basename(kwargs['wall_mesh_file']).split('.')[0]

    out_file = open(fmf+"_"+wmf+".urdf",'w') # TODO

    for in_line in in_file:
        in_line = in_line.replace('${floor_mesh_file}', 
                                  str(kwargs['floor_mesh_file']))
        in_line = in_line.replace('${wall_mesh_file}',
                                  str(kwargs['wall_mesh_file']))
        #print(in_line, end='')
        out_file.write(in_line)
    out_file.close()
    in_file.close()





    # TODO


def process_xacro(xacro_path, **kwargs):
    #print('process_xaro called with:')
    #print('Xacro-Path: ', xacro_path)

    # insert the parameters in the xacro template
    args = [xacro_path]
    for key, val in kwargs.items():
        args.append(key + ":=" + str(val))
    #print(args)
    
    opts, input_file_name = xacro.process_args(args)
    for k,v in kwargs.items():
        print("key:",k,"val:",v)
    print("opts;",opts)

    #return
    ## TODO: somehow the replacing does not work correctly (Exception)
    #doc = xacro.process_file(input_file_name, **vars(opts))
    doc = xacro.process_file(input_file_name, mappings={})
    doc_string = doc.toprettyxml()
    #print(doc)

    ## TODO: see "xacroFinal.py" on how to save the xacros/urdfs
    out_path = '../data/objs/pieces/envs/urdf/'
    # Generate urdf-file in the designated folder
    urdf_file="wall_floor.urdf"
    urdf_out = out_path + urdf_file
    print("os cwd:",os.getcwd())
    print("URDF out:",urdf_out)
    save_xml(urdf_out, doc)
    os.system('rm ' + os.getcwd() + '/' + xacro_file) # cleaning up

    #print('\n')

def load_objs(path):
    objs = [path + file for file in os.listdir(path) if file.endswith('.obj')]
    return objs

def generate_env(xacro_path):

    # load paths to objects to use in the environment
    mesh_path = xacro_path.replace(os.path.basename(xacro_path), 'mesh/')
    floor_path = os.path.join(mesh_path, 'floor/')
    wall_path = os.path.join(mesh_path, 'wall/')

    floors = load_objs(floor_path)    
    walls = load_objs(wall_path)

    for f in floors:
        for w in walls:
            #process_xacro(xacro_path, floor_mesh_file=f, wall_mesh_file=w)
            process_xacro2(xacro_path, floor_mesh_file=f, wall_mesh_file=w)
            pass
    



def main():
    env_xacro_template = '../data/objs/pieces/envs/env.urdf'

    if os.path.exists(env_xacro_template):
        generate_env(env_xacro_template)
    else:
        print('Environment-Template not found')

if __name__ == '__main__':
    main()
