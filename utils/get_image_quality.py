"""
通过计算psnr来获取每张图片的质量
"""
import os
import logging

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm
from PIL import Image
import numpy as np

RESULT_IMG_PATH = "/home/bingxing2/ailab/group/ai4agr/jinzhel/protein_work_data/vision_data/result_6_25/seq_img_valid_17kind_one_label"
# RESULT_IMG_PATH = "/home/bingxing2/ailab/group/ai4agr/jinzhel/protein_work_data/vision_data/result_5_16/seq_img_valid"


def set_log(file_path):
    logger = logging.getLogger(file_path)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    return logger


def get_all_img(folder_path):
    entries = os.listdir(folder_path)
    file_names = [entry for entry in entries if os.path.isfile(os.path.join(folder_path, entry))]
    pre_ptp = []
    truth_ptp = []
    psnr_folder_log = set_log(file_path="/home/bingxing2/ailab/scxlab0065/Predict_Protein_Subcellular_Localization/utils/image_seq_valid_17kind_psnr.txt")
    ssim_folder_log = set_log(file_path="/home/bingxing2/ailab/scxlab0065/Predict_Protein_Subcellular_Localization/utils/image_seq_valid_17kind_ssim.txt")
    for file_name in file_names:
        name_without_dot = file_name.split('.')[0]
        parts = name_without_dot.split("_")
        if parts[1] == "pre":
            pre_ptp.append(file_name)
        elif parts[1] == "truth":
            truth_ptp.append(file_name)
    all_quality = 0
    protein_dict = {}
    # n_10, n_12, n_14, n_16, n_18 = 0, 0, 0, 0, 0
    for i, pre_item in enumerate(pre_ptp, start=0):
        truth_item = pre_item.replace("pre", "truth")
        print(pre_item, truth_item)
        # if truth_ptp[i].split('.')[0].split("_")[2:4] == pre_item.split('.')[0].split("_")[2:4]:
        truth_img = np.array(Image.open(os.path.join(RESULT_IMG_PATH, truth_item)))
        pre_img = np.array(Image.open(os.path.join(RESULT_IMG_PATH, pre_item)))
        # print(truth_img.shape, pre_img.shape)
        img_quality_psnr = peak_signal_noise_ratio(truth_img, pre_img)
        img_quality_ssim = structural_similarity(truth_img, pre_img, data_range=truth_img.max()-truth_img.min())

        psnr_folder_log.info(pre_item + ":" + str(img_quality_psnr))
        ssim_folder_log.info(pre_item + ":" + str(img_quality_ssim))


if __name__ == "__main__":
    get_all_img(RESULT_IMG_PATH)

