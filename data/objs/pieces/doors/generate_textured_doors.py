import os
import re
import bpy


# load one blend-file and aplly all possible textures to it
def main():
    blend_dir = os.path.abspath('./blend/')
    mesh_dir = os.path.abspath('./mesh/')
    # print(os.getcwd())
    # print('go to mesh-dir:')
    os.chdir(mesh_dir)
    # print(os.getcwd())
    img_types = ['jpg', 'png']
    textures = sorted([tex for tex in os.listdir(os.path.relpath(mesh_dir + '/tex/')) if tex.split('.')[-1] in img_types])
    # print(textures)
    
    blend_file_path = bpy.data.filepath 
    blend_file = blend_file_path.split('/')[-1]

    for texture in textures:
        i = re.search('[1-9][0-9]?', texture).group(0) # number of texture

        # get the object of the scene
        ob = bpy.context.scene.objects[0]

        # generate a new material
        mat_name = texture.split('.')[0] # get the filename
        mat = bpy.data.materials.new(name=mat_name)

        # link nodes in shader editor to apply textures
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        texImage = mat.node_tree.nodes.new('ShaderNodeTexImage') # Texture Node
        texImage.image = bpy.data.images.load('./tex/' + texture)
        mat.node_tree.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])

        # Assign it to object
        if ob.data.materials:
            # assign to 1st material slot
            ob.data.materials[0] = mat
        else:
            # no slots
            ob.data.materials.append(mat)

        # output a obj-file after applying the texture
        door_prefix = blend_file.split('.')[0]
        # print('door_prefix = ', door_prefix)
        target_file = door_prefix + '_' + i + '.obj'
        # print(target_file)
        bpy.ops.export_scene.obj(filepath=target_file, axis_up='Z', axis_forward='Y')




if __name__ == '__main__':
    main()