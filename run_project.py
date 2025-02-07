from paths import *
from helpers import *
import torch
from torchvision import transforms
import os


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def load_models():
    models = {"dad" : [], "deedee" : [], "dexter" : [], "mom" : [], "unknown" : []}

    for label in LABELS:
        for model_name in MODEL_NAMES[label]:
            model = torch.load(os.path.join(MODELS_PATH, model_name), weights_only= False, map_location=device)
            model.eval()
            models[label].append(model.to(device))
    return models

def compute_score(window, char):

    if char in GRAY:
        window_tensor = transforms.ToTensor()(cv.cvtColor(window, cv.COLOR_BGR2GRAY)).unsqueeze(0)
    else:
        window_tensor = transforms.ToTensor()(window).unsqueeze(0)

    score = 0

    if torch.cuda.is_available():
        window_tensor = window_tensor.cuda()
    else:
        window_tensor = window_tensor.cpu()

    for model in models[char]:
        model.eval()
        with torch.no_grad():
            pred = model(window_tensor)

            score += pred.item()

    return score / len(models[char])


imgs = import_images(VALIDATION_PATH + "*.jpg", True)

models = load_models()

step_size = 16


all_det = {"dad" : np.array([[0, 0, 0, 0]]), "deedee" : np.array([[0, 0, 0, 0]]), "dexter" : np.array([[0, 0, 0, 0]]), "mom": np.array([[0, 0, 0, 0]]), "unknown": np.array([[0, 0, 0, 0]])}
all_scores = {"dad" : np.array([]), "deedee" : np.array([]), "dexter" : np.array([]), "mom": np.array([]), "unknown": np.array([])}
all_image_names = {"dad" : np.array([]), "deedee" : np.array([]), "dexter" : np.array([]), "mom": np.array([]), "unknown": np.array([])}
for i in tqdm(range(len(imgs))):
    detections = {"dad" : [], "deedee" : [], "dexter" : [], "mom": [], "unknown": []}
    scores = {"dad" : [], "deedee" : [], "dexter" : [], "mom": [], "unknown": []}
    for scale in scales:
        new_width = int(imgs[i].shape[1] * scale)
        new_height = int(imgs[i].shape[0] * scale)
        img = cv.resize(imgs[i], (new_width, new_height))
        for label in LABELS:
            for y in range(0, img.shape[0] - WINDOW_SIZES[label][0] + 1, step_size):
                for x in range(0, img.shape[1] - WINDOW_SIZES[label][1] + 1, step_size):
                    score = compute_score(img[y:y + WINDOW_SIZES[label][0], x:x + WINDOW_SIZES[label][1]], label)
                    if score >= SCORE_THRESH[label]:
                        detections[label].append([int(x / scale), int(y / scale),int((x + WINDOW_SIZES[label][1]) / scale), int((y + WINDOW_SIZES[label][0]) / scale)])
                        scores[label].append(score)
    for label in LABELS:
        if label == "unknown":
            detections[label] = np.array([det for label in LABELS for det in detections[label]])
            scores[label] = np.array([s for label in LABELS for s in scores[label]])   
        if len(detections[label]) > 0:
            detections[label], scores[label] = non_maximal_suppression(np.array(detections[label]), np.array(scores[label]), imgs[i].shape)
            all_det[label] = np.concatenate((all_det[label], detections[label]))
            all_scores[label] = np.concatenate((all_scores[label], scores[label]))
            all_image_names[label] = np.concatenate((all_image_names[label], np.array([f"{(i + 1):03}.jpg" for _ in range(len(detections[label]))])))     

if not os.path.exists("352_Lutu_Adrian-Catalin"):
    os.mkdir("352_Lutu_Adrian-Catalin")

os.chdir(r"352_Lutu_Adrian-Catalin")

if not os.path.exists("task1"):
    os.mkdir("task1")

if not os.path.exists("task2"):
    os.mkdir("task2")

os.chdir(r"task1")

np.save("detections_all_faces", all_det["unknown"][1:])
np.save("file_names_all_faces", all_image_names["unknown"])
np.save("scores_all_faces", all_scores["unknown"])

os.chdir(os.path.join(r"..", r"task2"))

np.save("detections_dad", all_det["dad"][1:])
np.save("file_names_dad", all_image_names["dad"])
np.save("scores_dad", all_scores["dad"])

np.save("detections_mom", all_det["mom"][1:])
np.save("file_names_mom", all_image_names["mom"])
np.save("scores_mom", all_scores["mom"])

np.save("detections_dexter", all_det["dexter"][1:])
np.save("file_names_dexter", all_image_names["dexter"])
np.save("scores_dexter", all_scores["dexter"])

np.save("detections_deedee", all_det["deedee"][1:])
np.save("file_names_deedee", all_image_names["deedee"])
np.save("scores_deedee", all_scores["deedee"])



