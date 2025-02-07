PATH = "C:\\Users\\Adi\\Desktop\\CAVA-2024-TEMA2\\CAVA-2024-TEMA2\\antrenare\\"
PATH_ANNOTATIONS = "C:\\Users\\Adi\\Desktop\\CAVA-2024-TEMA2\\CAVA-2024-TEMA2\\antrenare\\"
NAMES = ["dad", "deedee", "dexter", "mom"]
LABELS = ["dad", "deedee", "dexter", "mom", "unknown"]
COLORS = {"dad" : (255, 0, 0), "deedee" : (0, 255, 0), "dexter" : (0, 0, 255), "mom": (255, 255, 0), "unknown": (255, 0, 255)}
FACES_PATH = "C:\\Users\\Adi\\Desktop\\CAVA-2024-TEMA2\\CAVA-2024-TEMA2\\faces\\"
BACKGROUND_PATH = "D:\\DexterData\\42kScaled\\"
VALIDATION_PATH = "C:\\Users\\Adi\\Desktop\\CAVA-2024-TEMA2\\validare\\validare\\"
MODELS_PATH = "Models"
MODEL_NAMES = {"dad" : ["CNNDad5straturiGray14E42kbackground.pth"],
               "deedee" : ["CNNDeeDee5straturi11E42kbackground.pth"],
               "dexter" : ["CNNDexter5straturi10E42kbackground.pth"],
               "mom" : ["CNNMom5straturiGray20E42kbackground.pth"],
               "unknown" : ["CNNAllfaces4straturi1442k background.pth", "CNNAllfaces4straturi16E63kbackground.pth"]}
GRAY = set(["mom", "dad"])
WINDOW_SIZES = {"dad" : (143, 121), "deedee" : (114, 169), "dexter" : (111, 139), "mom": (114, 110), "unknown": (90, 92)}
SCORE_THRESH = {"dad" : 0.99, "deedee" : 0.99, "dexter" : 0.9, "mom": 0.99, "unknown": 0.99}
scales = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5]
NUM_IMG = 63000
TRAIN_IMG = 1000
