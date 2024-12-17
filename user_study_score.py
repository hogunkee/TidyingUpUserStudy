import copy
import os
import numpy as np
import cv2
import platform
from matplotlib import pyplot as plt


pf = platform.platform()
if pf.startswith('mac'):
    # For IOS
    KEY_1 = 49
    KEY_2 = 50
    KEY_3 = 51
    KEY_4 = 52
    KEY_5 = 53
    KEY_ENTER = 13
    KEY_BACKSPACE = 127
    KEY_ESC = 27
else:
    # For Ubuntu
    KEY_1 = 49
    KEY_2 = 50
    KEY_3 = 51
    KEY_4 = 52
    KEY_5 = 53
    KEY_ENTER = 13
    KEY_BACKSPACE = 8
    KEY_ESC = 27


def evaluate(data_folder, output_path, name):
    scenes = sorted([s for s in os.listdir(data_folder) if 'img' in s and '.png' in s])
    scenes_02 = [s for s in scenes if s.startswith("Score02")]
    scenes_24 = [s for s in scenes if s.startswith("Score24")]
    scenes_46 = [s for s in scenes if s.startswith("Score46")]
    scenes_68 = [s for s in scenes if s.startswith("Score68")]
    scenes_81 = [s for s in scenes if s.startswith("Score81")]

    scenes_list = [scenes_02, scenes_24, scenes_46, scenes_68, scenes_81]
    scenes = []
    for si, _scenes in enumerate(scenes_list):
        selected_scenes = np.random.choice(_scenes, 6, False)
        scenes += list(selected_scenes)
    np.random.shuffle(scenes)

    log_file = os.path.join(output_path, 'log_%s.txt'%name)
    with open(log_file, 'w') as file:
        file.write("Num scenes: %d\n" %len(scenes))

    scores = []
    logs = []

    sidx = 0
    while sidx < len(scenes):
        scene = scenes[sidx]
        #for sidx, scene in enumerate(scenes):
        print("    Current scene: [%d/%d]"%(sidx+1, len(scenes)))
        img_path = os.path.join(data_folder, scene)
        image = cv2.imread(img_path)
        cv2.imshow('Image Viewer', image.astype(np.uint8))

        # Rate Each Scene
        score = -1
        key = cv2.waitKey(0)
        print(key)
        if key==KEY_1:
            score = 1
        elif key==KEY_2:
            score = 2
        elif key==KEY_3:
            score = 3
        elif key==KEY_4:
            score = 4
        elif key==KEY_5:
            score = 5
        elif key==KEY_ESC:
            cv2.destroyAllWindows()
            with open(log_file, 'a') as file:
                for lg in logs:
                    file.write(lg)
            return logs, scores 
        elif key==KEY_BACKSPACE:
            if sidx>0:
                sidx -= 2
        sidx += 1

        if key==KEY_BACKSPACE:
            logs.pop(-1)
            scores.pop(-1)
        else:
            logs.append("Scene %d: %s / %d\n" %(sidx, scene, score))
            scores.append(score)
            print("    Rating:", score)

    with open(log_file, 'a') as file:
        for lg in logs:
            file.write(lg)
    return logs, scores 


if __name__=='__main__':
    folder_path = 'data/'
    while True:
        name = input("\n    Name: ").replace(' ', '')
        print()
        output_path = 'logs/%s-S' %name
        if os.path.isdir(output_path):
            print("    Same Name Already Exists!! Use another name.")
        else:
            os.makedirs(output_path)
            break

    intro = """    You have to tidy up the scenes with minimal movements!

    Press 1~5 to rate the scenes.
    Press BACKSPACE to go back to the previous scene.

    """
    print(intro)
    input("    Start? ")
    print()

    log_transforms, log_images = evaluate(folder_path, output_path, name)
    print("\n    Finished.")
    print("\n    Please compress your log folder 'logs/%s' and send it to me."%name)
    print("\n    Thank you!")
