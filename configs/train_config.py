import argparse
import os
import numpy as np

# --- CẤU HÌNH PATH ---
DATA_ROOT = '/root/data'  # Thay cho /kaggle/input
CODE_ROOT = '.'           # Thay cho /kaggle/working/FIFO_impr

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
BETA = 0.005
BATCH_SIZE = 4
ITER_SIZE = 1
NUM_WORKERS = 4

# 1. Dataset Paths (Trỏ vào folder con đã chuẩn hóa)
DATA_DIRECTORY = os.path.join(DATA_ROOT, 'Cityscapes')
DATA_DIRECTORY_CWSF = os.path.join(DATA_ROOT, 'Foggy_Cityscapes')
DATA_DIR_RF = DATA_ROOT # Foggy Zurich root

# 2. List Paths (Nằm trong thư mục code)
DATA_LIST_PATH = f'{CODE_ROOT}/dataset/cityscapes_list/train_foggy_{BETA}.txt'
DATA_CITY_PATH = f'{CODE_ROOT}/dataset/cityscapes_list/clear_lindau.txt'
DATA_LIST_PATH_CWSF = f'{CODE_ROOT}/dataset/cityscapes_list/train_origin.txt'

# 3. Foggy Zurich List (Nằm trong data)
DATA_LIST_RF = os.path.join(DATA_ROOT, 'Foggy_Zurich/lists_file_names/RGB_light_filenames.txt')

INPUT_SIZE = '2048,1024'
INPUT_SIZE_RF = '1920,1080'
NUM_CLASSES = 19 
NUM_STEPS = 100000 
NUM_STEPS_STOP = 60000 
RANDOM_SEED = 1234
RESTORE_FROM = 'no_model'
RESTORE_FROM_fogpass = 'no_model'
SAVE_PRED_EVERY = 100
SNAPSHOT_DIR = f'{CODE_ROOT}/snapshots/FIFO_model' 
SET = 'train'

def get_arguments():

    parser = argparse.ArgumentParser(description="FIFO framework")

    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY)
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH)
    parser.add_argument("--data-city-list", type=str, default = DATA_CITY_PATH)
    parser.add_argument("--data-list-rf", type=str, default=DATA_LIST_RF)    
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE)
    parser.add_argument("--input-size-rf", type=str, default=INPUT_SIZE_RF)
    parser.add_argument("--data-dir-cwsf", type=str, default=DATA_DIRECTORY_CWSF)
    parser.add_argument("--data-list-cwsf", type=str, default=DATA_LIST_PATH_CWSF)
    parser.add_argument("--data-dir-rf", type=str, default=DATA_DIR_RF)
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES)
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM)
    parser.add_argument("--restore-from-fogpass", type=str, default=RESTORE_FROM_fogpass)
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY)
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--set", type=str, default=SET)
    parser.add_argument("--lambda-fsm", type=float, default=0.0000001)
    parser.add_argument("--lambda-con", type=float, default=0.0001)
    parser.add_argument("--lambda-boundary", type=float, default=0.1, help="Weight for boundary detection loss")
    parser.add_argument("--accum-steps", type=int, default=1, help="Number of gradient accumulation steps (effective batch size = batch_size * accum_steps)")
    parser.add_argument("--file-name", type=str, required=True)
    parser.add_argument("--modeltrain", type=str, required=True)
    return parser.parse_args()

args = get_arguments()