import os
import torch
from torch.utils.data import Dataset
import numpy as np
import openvino as ov
import nncf
import cv2
from easyocr.imgproc import resize_aspect_ratio, normalizeMeanVariance
import argparse

class ICDAR2015Dataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('jpg', 'png'))])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load and preprocess an image
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image, _, _ = resize_aspect_ratio(image, 2560,  interpolation=cv2.INTER_LINEAR)
        image = np.transpose(normalizeMeanVariance(image), (2, 0, 1))
        return image

def transform_fn(data_item):
    image = data_item
    return image
    
def quantize_craft(model_path, image_dir):
    core = ov.Core()
    model = core.read_model(model_path) 
    
    #Prepare a calibration dataset
    dataset = ICDAR2015Dataset(image_dir)
    val_data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    calibration_dataset = nncf.Dataset(val_data_loader, transform_fn)
    
    # Run quantization and save INT8 model
    ov_quantized_model = nncf.quantize(model, calibration_dataset)
    int8_model_path="./INT8/"+model_path.replace(".xml", "_int8.xml")
    ov.save_model(ov_quantized_model, int8_model_path, compress_to_fp16=False)
    print("INT8 model is saved to", int8_model_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help = "Path to the model")
    parser.add_argument("image_dir", help = "Path to the directory with images")
    args = parser.parse_args()
    quantize_craft(args.model_path, args.image_dir)


if __name__ == "__main__":
    main()