import torch
from Utils.dataset import LyftDataset, preprocess
from config import *
from Utils.models import UNet
import argparse
from tqdm import tqdm


def load_model(model_path):
    model = UNet(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path))
    model = model.to(DEVICE)
    model.eval()
    # print(f"model summery: {model}")

    return model


def load_dataset(images_dir, masks_dir):
    test_dataset = LyftDataset(images_dir, masks_dir, transform=preprocess)
    test_iterator = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    return test_iterator


def check_accuracy(loader, model):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in tqdm(loader, total=len(loader)):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            softmax = torch.nn.Softmax(dim=1)
            preds = torch.argmax(softmax(model(x)), axis=1)
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    accuracy = num_correct / num_pixels
    accuracy = round(accuracy.item() * 100, 2)
    final_dice_score = dice_score / len(loader)

    print(f"Got {num_correct}/{num_pixels} with acc {accuracy}")
    print(f"Dice score: {final_dice_score.item()}")
    model.train()





# for x,y in test_iterator:
#     x = x.to(DEVICE)
#     fig , ax =  plt.subplots(3, 3, figsize=(18, 18))
#     softmax = torch.nn.Softmax(dim=1)
#     preds = torch.argmax(softmax(model(x)),axis=1).to('cpu')
#     img1 = np.transpose(np.array(x[0,:,:,:].to('cpu')),(1,2,0))
#     preds1 = np.array(preds[0,:,:])
#     mask1 = np.array(y[0,:,:])
#     img2 = np.transpose(np.array(x[1,:,:,:].to('cpu')),(1,2,0))
#     preds2 = np.array(preds[1,:,:])
#     mask2 = np.array(y[1,:,:])
#     img3 = np.transpose(np.array(x[2,:,:,:].to('cpu')),(1,2,0))
#     preds3 = np.array(preds[2,:,:])
#     mask3 = np.array(y[2,:,:])
#     ax[0,0].set_title('Image')
#     ax[0,1].set_title('Prediction')
#     ax[0,2].set_title('Mask')
#     ax[1,0].set_title('Image')
#     ax[1,1].set_title('Prediction')
#     ax[1,2].set_title('Mask')
#     ax[2,0].set_title('Image')
#     ax[2,1].set_title('Prediction')
#     ax[2,2].set_title('Mask')
#     ax[0][0].axis("off")
#     ax[1][0].axis("off")
#     ax[2][0].axis("off")
#     ax[0][1].axis("off")
#     ax[1][1].axis("off")
#     ax[2][1].axis("off")
#     ax[0][2].axis("off")
#     ax[1][2].axis("off")
#     ax[2][2].axis("off")
#     ax[0][0].imshow(img1)
#     ax[0][1].imshow(preds1)
#     ax[0][2].imshow(mask1)
#     ax[1][0].imshow(img2)
#     ax[1][1].imshow(preds2)
#     ax[1][2].imshow(mask2)
#     ax[2][0].imshow(img3)
#     ax[2][1].imshow(preds3)
#     ax[2][2].imshow(mask3)
#     break

# fig_path = os.path.join(ARTIFACTS_OUTPUT, "test_results.jpg")
# plt.savefig(fig_path, facecolor='w', transparent=False, bbox_inches='tight', dpi=100)
# plt.show()


def main(args):
    model = load_model(args.weights)
    dataset_iterator = load_dataset(args.images_dir, args.masks_dir)
    print("start calculating accuracy!")
    check_accuracy(dataset_iterator, model)



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='UNet.pt', help='model path')
    parser.add_argument('--images_dir', type=str, default='', help='dataset images dir path')
    parser.add_argument('--masks_dir', type=str, default='', help='dataset masks dir path')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    print(opt)
    main(opt)
