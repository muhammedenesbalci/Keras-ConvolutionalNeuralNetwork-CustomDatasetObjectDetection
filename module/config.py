import os

BASE_PATH = "C:\\Users\\enes\\Desktop\\objectDetection"

DATASET_PATH = BASE_PATH + "\\" + "dataset" + "\\" + "normal" 

IMAGES_PATH = DATASET_PATH + "\\" + "img"
IMAGES_ANNOATIONS_CSV_PATH = DATASET_PATH + "\\" + "csv_xml" + "\\" + "img_csv.csv"

BASE_OUTPUT_PATH = BASE_PATH + "\\" + "output"
MODEL_OUTPUT_PATH = BASE_OUTPUT_PATH  + "\\" + "detector_model.h5"
TEST_OUTPUT_FILENAMES = BASE_OUTPUT_PATH + "\\" + "test_images_txt"


INIT_LR = 1e-4
NUM_EPOCHS = 10
BATCH_SIZE = 24

