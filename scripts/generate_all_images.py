import image_generation
import sys
import os
import random
import itertools

def main(samples):
    env_path="../data/objs/generated_objs/generated_envs/urdf"
    door_path = "../data/objs/generated_objs/generated_doors/urdf"
    bright_walls = ["wall_white", "wall_concrete"]
    bright_doors = ["_6_dos", "_7_dos", "_9_dos", "_11_dos", "_6_cus", "_7_cus", "_9_cus", "_11_cus", "_6_cas",
                    "_7_cas", "_9_cas", "_11_cas"]
    env_urdfs = [f for f in os.listdir(env_path) if f.endswith('.urdf')]
    door_urdfs = [f for f in os.listdir(door_path) if f.endswith('.urdf')]
    combinations = list(itertools.product(env_urdfs,door_urdfs))
    random.shuffle(combinations)
    for combination in combinations:
        if samples==0:
            break
        env_urdf=combination[0]
        door_urdf=combination[1]
        if any(x in env_urdf for x in bright_walls) and any(x in door_urdf for x in bright_doors):
            print("Skipped: ", env_urdf, " with: ", door_urdf)
            continue
        samples-=1
        print("python image_generation.py "+str(door_path+"/"+door_urdf)+" "+str(env_path+"/"+env_urdf))
        os.system("python image_generation.py "+str(door_path+"/"+door_urdf)+" "+str(env_path+"/"+env_urdf))


if __name__ == '__main__':
    samples = int(sys.argv[1])
    main(samples)