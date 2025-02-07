from helpers import *
import os
from paths import *

annotations = {"dad" : [[] for _ in range(TRAIN_IMG)], "deedee" : [[] for _ in range(TRAIN_IMG)], "dexter" : [[] for _ in range(TRAIN_IMG)], "mom": [[] for _ in range(TRAIN_IMG)], "unknown": [[] for _ in range(TRAIN_IMG)]}
images = {"dad" : [], "deedee" : [], "dexter" : [], "mom": []}

for name in NAMES:
        imgs = import_images(PATH + name + "\\*.jpg", True)
        images[name] = imgs
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
            if label == name:
                annotations[label][img_no - 1].append((x_min, y_min, x_max, y_max))


imgs_per_scale = 0
for label in LABELS:
    no_img = 0
    back = []
    scale = 0.5
    while no_img < NUM_IMG:
          for name in NAMES:
                for i in range(len(images[name])):
                        if no_img >= NUM_IMG:
                            continue
                        img = images[name][i]
                        if scale != 1:
                            new_width = int(img.shape[1] * scale)
                            new_height = int(img.shape[0] * scale)
                            img = cv.resize(img, (new_width, new_height))
                        
                        num_rows = img.shape[0]
                        num_cols = img.shape[1]
                        x = np.random.randint(low = 0, high = num_cols - WINDOW_SIZES[label][1])
                        y = np.random.randint(low = 0, high = num_rows - WINDOW_SIZES[label][0])

                        is_ok = True
                        if label != "unknown":
                            for ann in annotations[label][i]:
                                score = intersection_over_union(ann, [int(x / scale), int(y / scale), int((x + WINDOW_SIZES[label][1]) / scale), int((y + WINDOW_SIZES[label][0]) / scale)])
                                if score >= 0.5:
                                    is_ok = False
                        else:
                            for l in LABELS:
                                for ann in annotations[l][i]:
                                    score = intersection_over_union(ann, [int(x / scale), int(y / scale), int((x + WINDOW_SIZES[label][1]) / scale), int((y + WINDOW_SIZES[label][0]) / scale)])
                                    if score >= 0.5:
                                        is_ok = False
                        
                        if is_ok and no_img < NUM_IMG:
                            back.append(img[y:y + WINDOW_SIZES[label][0], x:x + WINDOW_SIZES[label][1]].copy())
                            no_img += 1
                            imgs_per_scale += 1
                            if imgs_per_scale == 3000:
                                print(f"{scale} done")
                                scale += 0.1
                                imgs_per_scale = 0

    if not os.path.exists(BACKGROUND_PATH + label + "_background"):
        print(label, len(back))
        np.save(BACKGROUND_PATH + label + "_background", back)
