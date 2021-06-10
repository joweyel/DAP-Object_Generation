import os
import re
import json
import bpy

f = lambda x: int(x.split('_')[-1].split('.')[0]) # for more natural sorting

def generate_obj(json_data, a):
    pass

# load one blend-file and aplly all possible textures to it
def main():
    # go to the relevant directories
    blend_dir = os.path.abspath('./blend/')
    mesh_dir = os.path.abspath('./mesh/')
    os.chdir(mesh_dir)
    tex_path = os.path.join(mesh_dir, 'tex')


    # get type of the door {door, cupboard, cabinet}
    blend_file_path = bpy.data.filepath 
    door_nr = int(re.search('[1-9][0-9]?', blend_file_path).group(0))


    # door_type = os.path.basename(blend_file_path).split('_')[0]
    door_info = bpy.context.scene.objects[0].data.name
    door_types = door_info.split('_')
    print('type: ', door_types, ' | nr = ', door_nr)


    # process needed information
    json_path = '../../door.json'

    with open(json_path,) as file:
        data = json.load(file) 
    
    for door_type in door_types:
        if door_nr not in data['doors'][door_type]['objects']:
            data['doors'][door_type]['objects'].append(door_nr)
        else:
            print('Door {} already present!'.format(door_nr))
    
    # save the updateted json-data
    with open(json_path, "w") as file:
        json.dump(data, file, indent=4)


    # get all the relevant texture numbers (all these are used; some of the maybe
    # for more than one door type)
    l = []
    for door_type in door_types:
        l.extend(data['doors'][door_type]['texture'])
    tex_nr = list(set(l))
    textures = ['tex_' + str(i) + '.png' for i in tex_nr]
    textures = sorted([tex for tex in textures if tex in os.listdir(os.path.relpath(tex_path))], key=f)
    print(textures)

    blend_file = os.path.basename(blend_file_path)

    for texture in textures:
        i = re.search('[1-9][0-9]?', texture).group(0) # number of texture
        ob = bpy.context.scene.objects[0]              # get the object of the scene

        # generate a new material
        mat_name = texture.split('.')[0] # get the filename
        mat = bpy.data.materials.new(name=mat_name)

        # link nodes in shader editor to apply textures
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        texImage = mat.node_tree.nodes.new('ShaderNodeTexImage') # Texture Node
        texImage.image = bpy.data.images.load(os.path.join('./tex', texture))
        mat.node_tree.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])

        if ob.data.materials:  # Application of material
            ob.data.materials[0] = mat
        else:
            ob.data.materials.append(mat)

        # output a obj-file after applying the texture
        door_prefix = blend_file.split('.')[0]
        target_file = door_prefix + '_' + i + '.obj'
        bpy.ops.export_scene.obj(filepath=target_file, axis_up='Z', axis_forward='Y')



if __name__ == '__main__':
    main()