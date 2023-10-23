import os
from glob import glob
import torch
from PIL import Image
from Utils.models import UNet 
from Utils.dataset import preprocess
import torch.nn.functional as F
from config import *
from Utils.utils import *
import argparse 



def load_model(model_path):
    model = UNet(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path))
    model = model.to(DEVICE)
    model.eval()
    # print(f"model summery: {model}")

    return model


def predict(image, model):
    with torch.no_grad():
        image = image.to(DEVICE)
        y_pred = model(image)
        y_prob = F.softmax(y_pred, dim=1)
        top_pred = y_prob.argmax(1, keepdim=True)
        top_pred = torch.squeeze(top_pred)
    
    return top_pred


def main(args):
    model = load_model(args.weights)
    image = Image.open(args.img)
    image_arr = np.array(image)
    transform = preprocess(image=image_arr)
    processed_image = transform["image"]
    processed_image = processed_image.unsqueeze(0)
    prediction = predict(processed_image, model)
    prediction = prediction.detach().cpu().numpy()
    # Visualize example
    visualize(image, prediction, fig_name="predict_result.jpg", save=True)




def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='UNet.pt', help='model path')
    parser.add_argument('--img', type=str, default='test.png', help='image file path')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    print(opt)
    main(opt)
    