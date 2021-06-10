import os
import re
import json
import bpy

f = lambda x: int(x.split('_')[-1].split('.')[0]) # for more natural sorting

# load one blend-file and aplly all possible textures to it
def main():

    blend_dir = os.path.abspath('./blend/')
    mesh_dir = os.path.abspath('./mesh/')
    os.chdir(mesh_dir)
    tex_path = os.path.join(mesh_dir, 'tex')

    # path to blend-file
    blend_file_path = bpy.data.filepath
    
    # get number of handle
    handle_nr = re.search('[1-9][0-9]?', blend_file_path).group(0)
    
    # get type of the door-handle {handle, knob, bar}
    handle_type = bpy.context.scene.objects[0].data.name
    print('type: ', handle_type, ' | nr = ', handle_nr)    

    # open json to get the number of permitted textures
    json_path = '../../door.json'
    
    with open(json_path,) as file:
        data = json.load(file)


    # add object number to the corresponding door-handle
    if int(handle_nr) not in data['handles'][handle_type]['objects']:
        data['handles'][handle_type]['objects'].append(int(handle_nr))
    else:
        print('Handle {} already present!'.format(handle_nr))
        
    with open(json_path, "w") as file:
        json.dump(data, file, indent=4)

    # get all the textures
    tex_nr = data['handles'][handle_type]['texture']
    textures = ['tex_' + str(i) + '.png' for i in tex_nr]
    textures = sorted([tex for tex in textures if tex in os.listdir(os.path.relpath(tex_path))], key=f)
    print(textures)

    blend_file = os.path.basename(blend_file_path)

    for texture in textures:

        i = re.search('[1-9][0-9]?', texture).group(0) # number of texture
        ob = bpy.context.scene.objects[0]              # get the object of the scene

        # generate a new material
        mat_name = texture.split('.')[0]
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
        bpy.ops.export_scene.obj(filepath=target_file, axis_up='X', axis_forward='Y')




if __name__ == '__main__':
    main()