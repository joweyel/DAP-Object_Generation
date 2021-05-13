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

def generate_xacro(xacro_path, **kwargs):

    # process the inputs and insert them into the xacro document
    args = [xacro_path]
    for key, value in kwargs.items():
        args.append(key + ":=" + str(value))
    opts, input_file_name = xacro.process_args(args)
    doc = xacro.process_file(input_file_name, **vars(opts))

    # output path where the xacro will be saved to
    xacro_output = kwargs['mesh_file'].split('/')[-1].replace('.obj', '.xacro')
    out_path = "/".join(xacro_path.split("/")[:-1]) + '/xacro/' + xacro_output
    print('Saving xacro-file at: ', out_path)
    save_xml(out_path, doc)


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
            # mesh = "$(find pieces)/doors/mesh/" + obj_type + str(obj_path.split('_')[-1]) + '.obj'    # works only with ros
            print(mesh)
            generate_xacro(xacro_path, mesh_file=mesh, parent='world',
                           size_x='1.0', size_y='1.0', size_z='1.0')
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