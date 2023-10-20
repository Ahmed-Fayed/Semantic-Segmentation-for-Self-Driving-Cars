import os
from glob import glob
import torch
from PIL import Image
from Utils.models import UNet  #, TransUNet
from Utils.dataset import LyftDataset, t1
import torch.nn.functional as F
from config import *
from Utils.utils import *


model = UNet(num_classes=13)
# model = TransUNet(n_channels=3, n_classes=6)
model.load_state_dict(torch.load(model_name))
print(f"model summery: {model}")

model.eval()

images_dir = BASE_OUTPUT + "/test_dataset/images"
masks_dir = BASE_OUTPUT + "/test_dataset/masks"
test_dataset = LyftDataset(images_dir, masks_dir, transform=t1)
test_iterator = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
images, masks = next(iter(test_iterator))

original_images = glob(images_dir + "/*.png")

with torch.no_grad():
    y_pred = model(images)
    y_prob = F.softmax(y_pred, dim=1)
    top_pred = y_prob.argmax(1, keepdim=True)
    top_pred = torch.squeeze(top_pred)
    print(f"mask: {top_pred}, shape: {top_pred.shape}")

img_0 = Image.open(original_images[0])

# Visualize example
display(img_0, masks[0], top_pred[0], "Original Image", "Predicted Mask", "Final_Prediction_2", save=False)


def check_accuracy(loader, model):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            # x = x.to(DEVICE)
            # y = y.to(DEVICE)
            softmax = torch.nn.Softmax(dim=1)
            preds = torch.argmax(softmax(model(x)), axis=1)
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()


# check_accuracy(test_iterator, model)


for x,y in test_iterator:
    # x = x.to(DEVICE)
    fig , ax =  plt.subplots(3, 3, figsize=(18, 18))
    softmax = torch.nn.Softmax(dim=1)
    preds = torch.argmax(softmax(model(x)),axis=1).to('cpu')
    img1 = np.transpose(np.array(x[0,:,:,:].to('cpu')),(1,2,0))
    preds1 = np.array(preds[0,:,:])
    mask1 = np.array(y[0,:,:])
    img2 = np.transpose(np.array(x[1,:,:,:].to('cpu')),(1,2,0))
    preds2 = np.array(preds[1,:,:])
    mask2 = np.array(y[1,:,:])
    img3 = np.transpose(np.array(x[2,:,:,:].to('cpu')),(1,2,0))
    preds3 = np.array(preds[2,:,:])
    mask3 = np.array(y[2,:,:])
    ax[0,0].set_title('Image')
    ax[0,1].set_title('Prediction')
    ax[0,2].set_title('Mask')
    ax[1,0].set_title('Image')
    ax[1,1].set_title('Prediction')
    ax[1,2].set_title('Mask')
    ax[2,0].set_title('Image')
    ax[2,1].set_title('Prediction')
    ax[2,2].set_title('Mask')
    ax[0][0].axis("off")
    ax[1][0].axis("off")
    ax[2][0].axis("off")
    ax[0][1].axis("off")
    ax[1][1].axis("off")
    ax[2][1].axis("off")
    ax[0][2].axis("off")
    ax[1][2].axis("off")
    ax[2][2].axis("off")
    ax[0][0].imshow(img1)
    ax[0][1].imshow(preds1)
    ax[0][2].imshow(mask1)
    ax[1][0].imshow(img2)
    ax[1][1].imshow(preds2)
    ax[1][2].imshow(mask2)
    ax[2][0].imshow(img3)
    ax[2][1].imshow(preds3)
    ax[2][2].imshow(mask3)
    break

fig_path = os.path.join(ARTIFACTS_OUTPUT, "test_results.jpg")
# plt.savefig(fig_path, facecolor='w', transparent=False, bbox_inches='tight', dpi=100)
plt.show()
