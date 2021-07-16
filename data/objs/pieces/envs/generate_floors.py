import os
import re
import json
import bpy

def main():
    blend_dir = os.path.abspath('./blend/')
    mesh_dir = os.path.abspath('./mesh/')
    os.chdir(mesh_dir)
    tex_path = os.path.join(mesh_dir, 'tex')

    blend_file_path = bpy.data.filepath
    textures = sorted([tex for tex in os.listdir(tex_path) if tex.endswith('png')])

    for idx, texture in enumerate(textures):
        ob = bpy.context.scene.objects[0]
        mat_name = texture.split('.')[0]
        mat = bpy.data.materials.new(name=mat_name)
        
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        texImage = mat.node_tree.nodes.new('ShaderNodeTexImage') # Texture Node
        texImage.image = bpy.data.images.load(os.path.join('./tex', texture))
        mat.node_tree.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])
        
        if ob.data.materials:  # Application of material
            ob.data.materials[0] = mat
        else:
            ob.data.materials.append(mat)

        target_file = 'floor_' + str(idx + 1) + '.obj'
        bpy.ops.export_scene.obj(filepath=target_file, axis_up='Z', axis_forward='Y')
        print('Written [ ' + target_file + ' ]')

if __name__ == '__main__':
    main()