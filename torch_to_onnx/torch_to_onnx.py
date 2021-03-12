import cv2
import onnx
import onnxruntime as ort
import torch
import numpy as np
from albumentations import (Compose, Resize, )
from albumentations.augmentations.transforms import Normalize
from albumentations.pytorch.transforms import ToTensor
from torchvision import models
import time


def preprocess_image(img_path):
    # transformations for the input data
    transforms = Compose([
        Resize(224, 224, interpolation=cv2.INTER_NEAREST),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensor(),
    ])

    # read input image
    input_img = cv2.imread(img_path)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
    # do transformations
    input_data = transforms(image=input_img)["image"]
    # prepare batch
    batch_data = torch.unsqueeze(input_data, 0)

    return batch_data


def main():
    # load pre-trained model -------------------------------------------------------------------------------------------
    model = models.resnet18(pretrained=False)
    fc_features = model.fc.in_features
    model.fc = torch.nn.Linear(fc_features, 5)
    checkpoint = torch.load('model_best.pth.tar')
    model.load_state_dict({k.replace('module.', ''):
                               v for k, v in checkpoint['state_dict'].items()})
    # preprocessing stage ----------------------------------------------------------------------------------------------
    input = preprocess_image("turkish_coffee.jpg").cuda()
    test_arr = input.cpu().numpy().astype(np.float32)
    # inference stage --------------------------------------------------------------------------------------------------
    model.eval()
    model.cuda()

    torch.cuda.synchronize()
    start = time.time()
    for i in range(1000):
        output = model(input)
    torch.cuda.synchronize()
    print("pytorch time used ", (time.time() - start) / 1000)

    # post-processing stage --------------------------------------------------------------------------------------------
    print(output)
    # convert to ONNX --------------------------------------------------------------------------------------------------
    ONNX_FILE_PATH = "resnet18.onnx"
    torch.onnx.export(model, input, ONNX_FILE_PATH, input_names=["input"], output_names=["output"], export_params=True)
    onnx_model = onnx.load(ONNX_FILE_PATH)

    # check that the model converted fine
    ort_session = ort.InferenceSession('resnet18.onnx')
    torch.cuda.synchronize()
    start = time.time()
    for i in range(1000):
        outputs = ort_session.run(None, {'input': test_arr})
    torch.cuda.synchronize()
    print("onnx time used ", (time.time() - start) / 1000)
    print("output", outputs[0])
    onnx.checker.check_model(onnx_model)
    print("Model was successfully converted to ONNX format.")
    print("It was saved to", ONNX_FILE_PATH)


if __name__ == '__main__':
    main()
