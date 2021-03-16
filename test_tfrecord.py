# Download following dataset.
# gsutil -m cp -r gs://experiments_logs/gmm/TOPS/gl/dataset/generator_layers_v1.record /content/

import torch
from torch import nn, optim
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
# from tensorboardX import SummaryWriter
import tensorflow as tf

from PIL import Image
from tqdm import tqdm
from colorama import Fore

import os
import numpy as np

from utils import weights_init, unnormalize_batch
# from datasets import FashionGen, FashionAI, DeepFashion, DeepFashion2, CelebAHQ
from models import Net, PConvNet, Discriminator, VGG16
from losses import CustomLoss
import matplotlib.pyplot as plt

NUM_EPOCHS = 21
BATCH_SIZE = 16
save_path = "gs://vinit_helper/cloth_inpainting_gan/cloth_inpainting_eccv20_aim/"
exp_name = 'exp_1'

MODE = "train"
MASK_FORM = "free"  # "free"
MULTI_GPU = False
DEVICE_ID = 1
DEVICE = "cuda"

# MEAN = np.array([0.485, 0.456, 0.406])
# STD = np.array([0.229, 0.224, 0.225])


# mask_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
#                                          transforms.RandomVerticalFlip(p=0.5),
#                                          transforms.Resize(256),
#                                          transforms.ToTensor(), ])


device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

refine_net = Net()
refine_net.load_state_dict(torch.load("exp_mask_like_paper_v2/weights/weights_net_epoch_20.pth", map_location='cpu'))
# vgg = VGG16(requires_grad=False)
# vgg.to(device)

if torch.cuda.device_count() > 1 and MULTI_GPU:
    print("Using {} GPUs...".format(torch.cuda.device_count()))
    refine_net = nn.DataParallel(refine_net)
else:
    print("GPU ID: {}".format(device))

refine_net = refine_net.to(device)

d_loss_fn = nn.BCELoss()
d_loss_fn = d_loss_fn.to(device)
refine_loss_fn = CustomLoss()
refine_loss_fn = refine_loss_fn.to(device)

from dataset_tfrecord import define_dataset
tfrecord_path = "/content/generator_layers_v1.record"
batch_size = 1
trainset, trainset_length = define_dataset(tfrecord_path, batch_size, train=True)
valset, valset_length = define_dataset(tfrecord_path, batch_size, train=False)

# tps_weights_path = "gs://experiments_logs/gmm/TOPS/short_sleeves_high_slope_loss/weights/model_44"
# tps_model = tf.keras.models.load_model(tps_weights_path, custom_objects={"tf": tf}, compile=False)

def tensor2ndarray(image):
    return image.permute(0, 2, 3, 1).detach().cpu().numpy()

def train(epoch, results_folder):
    train_iterator = iter(trainset)
    num_iterations = int(trainset_length/batch_size)
    for batch_idx in range(num_iterations):
        num_step = epoch * trainset_length + batch_idx

        data, model_inputs = next(train_iterator)

        # what to add to the TPS cloth mask to make it complete?
        inpaint_region = data["inpaint_region"]

        person_cloth = data["person_cloth"]
        warped_cloth_input = model_inputs["warped_cloth"]

        x_train = torch.tensor(warped_cloth_input.numpy()).float().to(device)
        x_train = x_train.permute(0, 3, 1, 2)
        x_mask = inpaint_region.numpy()
        x_mask = torch.tensor(x_mask).float().to(device)
        x_mask = torch.cat([x_mask, x_mask, x_mask], dim=-1)
        x_mask = x_mask.permute(0, 3, 1, 2)
        x_mask = 1.0 - x_mask

        y_train = torch.tensor(person_cloth.numpy()).float().to(device)
        y_train = y_train.permute(0, 3, 1, 2)

        r_output = refine_net(x_train, x_mask)
        r_composite = x_mask * y_train + (1.0 - x_mask) * r_output

        x_mask = 1.0 - x_mask

        gt_cloth = (tensor2ndarray(y_train)[0]+1)*0.5
        input_cloth = (tensor2ndarray(x_train)[0]+1)*0.5
        gen_op = (tensor2ndarray(r_output)[0]+1)*0.5
        composite_op = (tensor2ndarray(r_composite)[0]+1)*0.5
        x_mask = tensor2ndarray(x_mask)[0]

        height, width = 256, 192
        result_image = np.zeros((2*height, 3*width, 3))
        result_image[:height, :width, :] = input_cloth
        result_image[:height, width:2*width, :] = gt_cloth
        result_image[:height, 2*width:3*width, :] = x_mask + (1-x_mask)*0.5

        result_image[height:2*height, :width, :] = input_cloth
        result_image[height:2*height, width:2*width, :] = gen_op
        result_image[height:2*height, 2*width:3*width, :] = composite_op

        plt.imsave(os.path.join(results_folder, '{}.jpg'.format(batch_idx)), result_image)

        # vgg_features_gt = vgg(y_train)
        # vgg_features_composite = vgg(r_composite)
        # vgg_features_output = vgg(r_output)

        # r_total_loss, r_pixel_loss, r_perceptual_loss, \
        #     adversarial_loss, r_tv_loss = refine_loss_fn(y_train, r_output, r_composite, d_output,
        #                                                  vgg_features_gt, vgg_features_output, vgg_features_composite)

if __name__ == '__main__':
    results_folder = 'results'
    if not os.path.exists(results_folder): os.mkdir(results_folder)
    if MODE == "train":
        print('Size of training datset: ', trainset_length)
        print('Training Started!')
        train(0, results_folder)