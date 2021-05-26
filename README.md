# Deep Articulation Prediction - Object Generation

## Convert xacro file to urdf file (executed from `src` directory)
```rosrun xacro xacro urdf_xacro/door_handle.urdf.xacro > urdf_xacro/out.urdf```

## Tasks: Week 01.05. - 07.05.
2. Create a [URDF](src/urdf_xacro/door.urdf) with a door link and a handle link and connect them with a fixed joint. Check in Pybullet if it works.<br>
`python3 object_examples.py urdf_xacro/door.urdf`

3. Create an indepedendent URDF for the Handles (Only a handle link and parameterize base_link).
See [door_handle.urdf](src/urdf_xacro/door_handle.urdf)
 for single door knob.<br>
 `python3 object_examples.py urdf_xacro/door_handle.urdf`

4. Try joining both door and handle through Xacro file(Puze is a professional in this). Relevant Files [Xacro](src/urdf_xacro/door_handle.urdf.xacro) & [URDF](src/urdf_xacro/out.urdf)<br>
`rosrun xacro xacro urdf_xacro/door_handle.urdf.xacro > urdf_xacro/out.urdf`<br>
`python3 object_examples.py urdf_xacro/out.urdf`

5. Use a Xacro to Urdf script (You can find it in the internet or ask puze) and create a unified URDF. Load it in Pybullet<br>
`rosrun xacro xacro urdf_xacro/door_handle.urdf.xacro > urdf_xacro/out.urdf`

For Fast changes to the **Xacro-File** and immediate execution use the following commant:<br>
`rosrun xacro xacro urdf_xacro/door_handle.urdf.xacro > urdf_xacro/out.urdf && python3 object_examples.py urdf_xacro/out.urdf`


## Generating Data (from the scripts folder)

[Hessenbox-Link](https://hessenbox.tu-darmstadt.de/getlink/MjZkWmh1aHo3QXFwaG0xVmg4NzlN/GIT_data) to bigger files like Blender-files and textures (to avoid tracking all of these files). These files have to be placed in the corrsponding folders.

### 1. Apply all possible textures to all possible Blender-files with the Python-API of Blender
[generate_textured_mesh.py](scripts/generate_textured_mesh.py) invokes a python script in the subdirectories of the doors and handles, where it calls the Blender-API for all possible combinations of Objects and Textures (currently only tested for the doors and not the handles)<br>
`python3 generate_textured_mesh.py`<br>
**Blender files have to be placed in**: `data/objs/pieces/{doors|handles}/blend/`<br>
**Textures have to be placed in**: `data/objs/pieces/{doors/handles}/mesh/tex/`<br><br>
[generate_textured_doors.py](data/objs/pieces/doors/generate_textured_doors.py) creates for each door **obj** & **mtl** of the signature `door_X_Y.{obj|mtl}` where `X=#door` and `Y=#texture`<br>(e.g. `door_2_9.obj` was created from `door_2.blend` and `tex_9.{png|jpg}`)<br><br>
[generate_textured_handles.py](data/objs/pieces/handles/generate_textured_handles.py) **TODO**


### 2. Generating the separate Xacros for doors and handles
In [xacroProcessor.py](scripts/xacroProcessor.py) the following values get assigned to the xacros
- **Doors**: mesh gets assigned, scales for (x, y, z) of the door get assigned
- **Handles**: mesh gets assigned

**Output-Format(door)**: `door_X_Y_sZ.xacro` with `X=#door`, `Y=#texture` and `Z=scale in [1.0, 2.0]`<br>
**Output-Folder**: `data/objs/pieces/{doors|handles}/xacro/`


### 3. Generating the Xacros for the whole door (door and handle)
[xacroDoor.py](scripts/xacroDoor.py) generates the Xacros, which "merge" the sole xacros of the door and handle together and saves them in `data/objs/generated_objs/generated_doors/xacro/`<br>
The following parameters get assigned to the generated Xacro:<br>
- **plane_xacro**: xacro of a certain door 
- **handle_xacro**: xacro of a certain hadle
- **handle_pos_{x|y|z}**: position of the handle (on the door)
- **handle_ori_{r|p|y}**: orientation of the handle (on the door)

**Output-Format**: `door_X_Y_sZ_hH.xacro` with `X=#door`, `Y=#texture`, `Z=scale in [1.0, 2.0]`, `H=#handle`<br>
**Output-Folder**: `data/objs/generated_objs/generated_doors/xacro/`

This script also converts the generated Xacro's to URDF's and places them in the designated folder `data/objs/generated_objs/generated_doors/urdf/`

**Output-Format**: `door_X_Y_sZ_hH.urdf`<br>
**Output-Folder**: `data/objs/generated_objs/generated_doors/urdf/`









## Important Links

### URDF
[URDF-Tutorials](http://wiki.ros.org/urdf/Tutorials)<br>
[URDF-Xarco Tutrial](https://www.youtube.com/playlist?list=PLK0b4e05LnzYpDnNeWJcQLju7JfJFX-lk)<br>
[URDF-Modeling](https://www.youtube.com/watch?v=UUwHK5ONTAQ)

### Blender
[Blender Tutorials 1](https://www.youtube.com/watch?v=bpvh-9H8S1g), [Blender Tutorials 2](https://www.youtube.com/watch?v=v6uBU5fgczE)<br>
[Blender (change Origin) 1](https://www.youtube.com/watch?v=_ojeeuNtJM8)<br>
[Blender (change Origin) 2](https://daler.github.io/blender-for-3d-printing/mesh_modeling/object-origin.html)<br>
[Blender (change Origin) 3](https://www.youtube.com/watch?v=-CiWNcPB1CY) (manually change the pos./pose of Coordinate Frame)
