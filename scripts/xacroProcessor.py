import os
from sys import settrace
import time
import argparse

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

    xacro_output = kwargs[str(obj_type) + '_mesh_file'].split('/')[-1].replace('.obj', '.xacro')
    out_path = "/".join(xacro_path.split("/")[:-1]) + '/xacro/' + xacro_output

    in_file = open(xacro_path, 'r')
    out_file = open(out_path, 'w')

    for in_line in in_file:
        if obj_type == 'door':
            in_line = in_line.replace("${size_x}", kwargs['size_x'])
            in_line = in_line.replace("${size_y}", kwargs['size_y'])
            in_line = in_line.replace("${size_z}", kwargs['size_z'])
        mesh = "${"+ str(obj_type) + "_mesh_file}"
        in_line = in_line.replace(mesh, kwargs[str(obj_type) + '_mesh_file'])
        # print(in_line, end='')
        out_file.write(in_line)


    return

    # doc = minidom.parseString(xacro_path)
    # edit_tag(doc, 'mesh', '', '')
    # # process the inputs and insert them into the xacro document
    # args = [xacro_path]
    # for key, value in kwargs.items():
    #     args.append(key + ":=" + str(value))
    # opts, input_file_name = xacro.process_args(args)
    # print(opts)
    # # doc = xacro.process_file(input_file_name, **vars(opts))
    # # output path where the xacro will be saved to
    # xacro_output = kwargs[str(obj_type) + '_mesh_file'].split('/')[-1].replace('.obj', '.xacro')
    # out_path = "/".join(xacro_path.split("/")[:-1]) + '/xacro/' + xacro_output
    # print('Saving xacro-file at: ', out_path)
    # save_xml(out_path, doc)


def main(input_args):
    obj_type = input_args.type[0]  # type of objects
    n = input_args.n               # number of object (specific number or all)

    # check in input_args, which directory to choose
    if obj_type == "door":
        xacro_path = os.path.abspath('../data/objs/pieces/' + obj_type + 's/plane.xacro')
    elif obj_type == "handle":
        xacro_path = os.path.abspath('../data/objs/pieces/' + obj_type + 's/handle.xacro')
    else:
        print('Invalid Object type specified. Aborting!')
        exit(-1)

    # load xacro file and generate new xacros
    print('Found xacro file in: ', xacro_path)

    # path to the objects
    mesh_path = '/'.join(xacro_path.split('/')[:-1]) + '/mesh/'
    avail_obj = sorted([mesh_path + obj for obj in os.listdir(mesh_path) if os.path.isdir(mesh_path + obj)])
    n_avail_obj = [int(number.split('_')[-1]) for number in avail_obj]

    # load one or all objects
    if n == 'all':
        for obj_path in avail_obj:
            mesh = obj_path + '/' + obj_type + str(obj_path.split('_')[-1]) + '.obj'
            mesh = os.path.relpath(mesh)
            print(mesh)
            if obj_type == 'door':
                print('door')
                generate_xacro(xacro_path, obj_type, door_mesh_file=mesh,
                               size_x='1.0', size_y='1.0', size_z='1.0')
            if obj_type == 'handle':
                print('handle')
                generate_xacro(xacro_path, obj_type, handle_mesh_file=mesh)
    elif int(n) in n_avail_obj:
        print(int(n))
        # TODO: implemetnation for single file
        pass
    else:
        print('Invalid number specified. Aborting!')
        exit(-1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Objs into Xacro files (with given parameters)')
    parser.add_argument('-type', type=str, nargs='+', help='Type of xacro file to use (door or handle)')
    parser.add_argument('-n', default='all', type=str, help='Number of object to process')

    input_args = parser.parse_args()
    main(input_args)