class Config(object):
    # train
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 50
    # LOAD_WEIGHTS_BEFORE_TRAINING = False
    # LOAD_WEIGHTS_FROM_EPOCH = 0
    #
    # TRAINING_CONFIG_NAME = "coco_w32_256x192"
    #
    # input image
    IMAGE_HEIGHT = 192
    IMAGE_WIDTH = 256
    CHANNELS = 3
    #
    # heatmap
    HEATMAP_WIDTH = 64
    HEATMAP_HEIGHT = 48
    SIGMA = 2
    #
    TRANSFORM_METHOD = "resize"   # random_crop, resize

    # dataset
    ROOT_PATH = "/home/oumingfeng/hrnetv1/"
    COCO_ROOT_DIR = ROOT_PATH+ "dataset/COCO/2017/"
    COCO_TRAIN_TXT = ROOT_PATH + "dataset/COCO/2017/coco_train.txt"
    COCO_VALID_TXT = ROOT_PATH + "dataset/COCO/2017/coco_valid.txt"

    num_of_joints = 17

    #
    # save model
    save_weights_dir = ROOT_PATH+ "saved_model/weights/"
    SAVE_FREQUENCY = 1
    #
    # # test
    # TEST_PICTURES_DIRS = ["", ""]  # "./experiment/xxx.jpg"
    # TEST_DURING_TRAINING = True
    # SAVE_TEST_RESULTS_DIR = "./experiment/"
    #
    # # color (r, g, b)
    # DYE_VAT = {"Pink": (255, 192, 203), "MediumVioletRed": (199, 21, 133), "Magenta": (255, 0, 255),
    #            "Purple": (128, 0, 128), "Blue": (0, 0, 255), "LightSkyBlue": (135, 206, 250),
    #            "Cyan": (0, 255, 255), "LightGreen": (144, 238, 144), "Green": (0, 128, 0),
    #            "Yellow": (255, 255, 0), "Gold": (255, 215, 0), "Orange": (255, 165, 0),
    #            "Red": (255, 0, 0), "LightCoral": (240, 128, 128), "DarkGray": (169, 169, 169)}
    #
    # def __init__(self):
    #     pass
    #
    # def get_dye_vat_bgr(self):
    #     bgr_color = {}
    #     for k, v in self.DYE_VAT.items():
    #         r, g, b = v[0], v[1], v[2]
    #         bgr_color[k] = (b, g, r)
    #     return bgr_color
    #
    # def color_pool(self):
    #     bgr_color_dict = self.get_dye_vat_bgr()
    #     bgr_color_pool = []
    #     for k, v in bgr_color_dict.items():
    #         bgr_color_pool.append(v)
    #     return bgr_color_pool
