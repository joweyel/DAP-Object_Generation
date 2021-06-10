import os
from natsort import natsorted

def main():
    root = os.getcwd()

    # get the directory with the data generation script (door/handle)
    door_dir = '../data/objs/pieces/doors/'
    print('Door dir: ', os.getcwd())
    os.chdir(door_dir)
    script = 'generate_textured_doors.py'
    print('Current dir: ', os.getcwd())
    blend_files = natsorted(os.listdir('./blend/'))
    print('blend_files: ', blend_files)

    for bf in blend_files:
        print('Door: ', bf)
        bash_query = 'blender -b ' + 'blend/' + bf + ' --python ' + script
        print(bash_query)
        os.system(bash_query)
        
    print('\n###############################################################\n')
    
    ## TODO: implementin the same thing for handles ##
    os.chdir(root) # back to the roots, namely the "scripts"-folder
    handle_dir = '../data/objs/pieces/handles'
    print('Current dir: ', os.getcwd())
    os.chdir(handle_dir)
    script = 'generate_textured_handles.py'
    print('Handle dir: ', os.getcwd())
    blend_files = natsorted([b for b in os.listdir('./blend/') if b.endswith('blend')])
    print('blend_files: ', blend_files)

    for bf in blend_files:
        print('Processing: ', bf)

        bash_query = 'blender -b ' + './blend/' + bf + ' --python ' + script
        os.system(bash_query)


if __name__ == '__main__':
    main()