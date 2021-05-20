import os
from sys import settrace
import time

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

def replace_value(doc, tag, value):
    '''
        doc: xml-document to parsed through
        tag: name of node, where parameter has to be altered
        value: value to be written onto the node-parameter
    '''
    nodes = doc.getElementsByTagName(tag)
    for node in nodes:
        print(node)


# def remove_tag_velocity_decay(doc):
#     nodes = doc.getElementsByTagName('velocity_decay')
#     nodes[0].parentNode.removeChild(nodes[0])
#     return doc

def load_doors(door_path, **kwargs):
    args = [os.path.join(os.getcwd(), door_path)] #

    for key, value in kwargs.items():
        args.append(key + ":=" + str(value))
    print(args)
    
    opts, input_file_name = xacro.process_args(args)
    print(opts, input_file_name)
    doc = xacro.process_file(input_file_name, **vars(opts))



def load_handles(handle_path, **kwargs):
    pass

def main():

    xacro_path = 'path/to/xacros/'
    xacro_path = os.path.join(os.getcwd(), 'plane.xacro')
    # load xacro file(s)
    print(xacro_path)
    load_doors(xacro_path, plane_file='obj/door2.obj', size_x=1.0, size_y=1.0, size_z=1.0)

if __name__ == '__main__':
    main()