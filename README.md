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
