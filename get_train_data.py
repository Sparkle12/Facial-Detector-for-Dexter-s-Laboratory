from helpers import *
import os
from paths import *


faces = {"dad" : [], "deedee" : [], "dexter" : [], "mom": [], "unknown": []}
mean_aspect_ratios = {"dad" : 0, "deedee" : 0, "dexter" : 0, "mom": 0, "unknown": 0}
number_of_examples = {"dad" : 0, "deedee" : 0, "dexter" : 0, "mom": 0, "unknown": 0}
mean_size = {"dad" : [0, 0], "deedee" : [0, 0], "dexter" : [0, 0], "mom": [0, 0], "unknown": [0, 0]}
train_size = {"dad" : (143, 121), "deedee" : (114, 169), "dexter" : (111, 139), "mom": (114, 110), "unknown": (90, 92)}

for name in NAMES:
        imgs = import_images(PATH + name + "\\*.jpg", True)
        f = open(PATH_ANNOTATIONS + name + "_annotations.txt")
        lines = f.readlines()
        f.close()

        for line in lines:
            comp = line.split()
            img_no = int(comp[0][:4])
            x_min = int(comp[1])
            y_min = int(comp[2])
            x_max = int(comp[3])
            y_max = int(comp[4])
            label = comp[5]

            faces[label].append(cv.resize(imgs[img_no - 1][y_min : y_max + 1, x_min: x_max + 1], (train_size[label][1], train_size[label][0])))
            mean_aspect_ratios[label] += (y_max - y_min)/(x_max- x_min)
            mean_size[label][0] += (y_max - y_min)
            mean_size[label][1] += (x_max - x_min)
            number_of_examples[label] += 1


for char in mean_aspect_ratios.keys():
    if not os.path.exists(FACES_PATH + char):
        np.save(FACES_PATH + char, faces[char])
    print(f"{char}\n Average Aspect Ratio: {mean_aspect_ratios[char]/number_of_examples[char]}\n Average Height: {mean_size[char][0] / number_of_examples[char]}\n Averave Width {mean_size[char][1] / number_of_examples[char]}")




