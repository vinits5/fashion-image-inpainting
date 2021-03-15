# Run the following code to download the dataset:
# gsutil -m cp -r gs://labelling-tools-data/tfrecords/uplara_tops/uplara_tops_v10_refined_grapy.record /content/

import torch
from torch import nn, optim
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
import tensorflow as tf

from PIL import Image
from tqdm import tqdm
from colorama import Fore

import os
import numpy as np

from utils import weights_init, unnormalize_batch
from datasets import FashionGen, FashionAI, DeepFashion, DeepFashion2, CelebAHQ
from models import Net, PConvNet, Discriminator, VGG16
from losses import CustomLoss

NUM_EPOCHS = 21
save_path = "gs://vinit_helper/cloth_inpainting_gan/cloth_inpainting_eccv20_aim"
exp_name = 'exp_mask_like_paper'

MODE = "train"
MASK_FORM = "free"  # "free"
MULTI_GPU = False
DEVICE_ID = 1
DEVICE = "cuda"
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


mask_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                         transforms.RandomVerticalFlip(p=0.5),
                                         transforms.Resize(256),
                                         transforms.ToTensor(), ])


device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

d_net = Discriminator()
refine_net = Net()

if MODE != "ablation":
    vgg = VGG16(requires_grad=False)
    vgg.to(device)

if torch.cuda.device_count() > 1 and MULTI_GPU:
    print("Using {} GPUs...".format(torch.cuda.device_count()))
    d_net = nn.DataParallel(d_net)
    refine_net = nn.DataParallel(refine_net)
else:
    print("GPU ID: {}".format(device))

d_net = d_net.to(device)
refine_net = refine_net.to(device)

if MODE == "train":
    refine_net.apply(weights_init)

d_loss_fn = nn.BCELoss()
d_loss_fn = d_loss_fn.to(device)
refine_loss_fn = CustomLoss()
refine_loss_fn = refine_loss_fn.to(device)

lr, r_lr, d_lr = 0.0004, 0.0004, 0.0004
d_optimizer = optim.Adam(d_net.parameters(), lr=d_lr, betas=(0.9, 0.999))
r_optimizer = optim.Adam(refine_net.parameters(), lr=r_lr, betas=(0.5, 0.999))

d_scheduler = optim.lr_scheduler.ExponentialLR(d_optimizer, gamma=0.9)
r_scheduler = optim.lr_scheduler.ExponentialLR(r_optimizer, gamma=0.9)

if MODE != "ablation":
    writer = SummaryWriter()

from dataset_cloth import define_dataset
tfrecord_path = "/content/uplara_tops_v10_refined_grapy.record"
batch_size = 8
trainset, trainset_length = define_dataset(tfrecord_path, batch_size, train=True)
valset, valset_length = define_dataset(tfrecord_path, batch_size, train=False)

tps_weights_path = "gs://experiments_logs/gmm/TOPS/short_sleeves_high_slope_loss/weights/model_44"
tps_model = tf.keras.models.load_model(tps_weights_path, custom_objects={"tf": tf}, compile=False)

def train(epoch):
    train_iterator = iter(trainset)
    num_iterations = int(trainset_length/batch_size)
    for batch_idx in range(num_iterations):
        num_step = epoch * trainset_length + batch_idx

        data, model_inputs = next(train_iterator)
        
        tps_inputs = {
            "cloth": data["cloth"], 
            "cloth_mask": data["cloth_mask"], 
            "gt_warped_mask": data["gt_warped_mask"], 
            "grid_img": data["grid_img"], 
            }
        tps_op = tps_model(tps_inputs, training=False)
        warped_cloth = tps_op["warped_cloth"]*data["occluded_cloth_mask"]
        tps_cloth_mask = tps_op["warped_mask"]*data["occluded_cloth_mask"]

        # warped_cloth, tps_cloth_mask = self.erode(warped_cloth, tps_cloth_mask) 

        # what to add to the TPS cloth mask to make it complete?
        inpaint_region = (tf.cast(tf.math.logical_or(tf.cast(data["occluded_cloth_mask"], tf.bool), tf.cast(tps_cloth_mask, tf.bool)), tf.float32) - tps_cloth_mask)

        person_cloth = data["person_cloth"]
        warped_cloth_input = person_cloth - person_cloth*inpaint_region
        model_inputs['warped_cloth'] = warped_cloth_input
        model_inputs['inpaint_region'] = inpaint_region
        data['inpaint_region'] = inpaint_region

        x_train = torch.tensor(warped_cloth_input.numpy()).float().to(device)
        x_train = x_train.permute(0, 3, 1, 2)
        x_mask = inpaint_region.numpy()
        x_mask = torch.tensor(x_mask).float().to(device)
        x_mask = torch.cat([x_mask, x_mask, x_mask], dim=-1)
        x_mask = x_mask.permute(0, 3, 1, 2)

        y_train = torch.tensor(person_cloth.numpy()).float().to(device)
        y_train = y_train.permute(0, 3, 1, 2)

        writer.add_scalar("LR/learning_rate", r_scheduler.get_lr(), num_step)

        refine_net.zero_grad()
        r_output = refine_net(x_train, x_mask)
        d_output = d_net(r_output.detach()).view(-1)
        r_composite = x_mask * r_output + (1.0 - x_mask) * y_train

        vgg_features_gt = vgg(y_train)
        vgg_features_composite = vgg(r_composite)
        vgg_features_output = vgg(r_output)

        r_total_loss, r_pixel_loss, r_perceptual_loss, \
            adversarial_loss, r_tv_loss = refine_loss_fn(y_train, r_output, r_composite, d_output,
                                                         vgg_features_gt, vgg_features_output, vgg_features_composite)

        writer.add_scalar("Refine_G/on_step_total_loss", r_total_loss.item(), num_step)
        writer.add_scalar("Refine_G/on_step_pixel_loss", r_pixel_loss.item(), num_step)
        writer.add_scalar("Refine_G/on_step_perceptual_loss", r_perceptual_loss.item(), num_step)
        writer.add_scalar("Refine_G/on_step_adversarial_loss", adversarial_loss.item(), num_step)
        writer.add_scalar("Refine_G/on_step_tv_loss", r_tv_loss.item(), num_step)

        if batch_idx%100 == 0:
            print(f"Epoch: {epoch+1} |  batch_idx: {batch_idx}/{num_iterations} -> total_loss: {r_total_loss.item()},\
                    pixel_loss: {r_pixel_loss.item()}, adv_loss: {adversarial_loss.item()}, perceptual_loss: {r_perceptual_loss.item()},\
                    tv_loss: {r_tv_loss.item()}")

        r_total_loss.backward()
        r_optimizer.step()

        d_net.zero_grad()
        d_real_output = d_net(y_train).view(-1)
        d_fake_output = d_output.detach()

        if torch.rand(1) > 0.1:
            d_real_loss = d_loss_fn(d_real_output, torch.FloatTensor(d_real_output.size(0)).uniform_(0.0, 0.3).to(device))
            d_fake_loss = d_loss_fn(d_fake_output, torch.FloatTensor(d_fake_output.size(0)).uniform_(0.7, 1.2).to(device))
        else:
            d_real_loss = d_loss_fn(d_real_output, torch.FloatTensor(d_fake_output.size(0)).uniform_(0.7, 1.2).to(device))
            d_fake_loss = d_loss_fn(d_fake_output, torch.FloatTensor(d_real_output.size(0)).uniform_(0.0, 0.3).to(device))

        writer.add_scalar("Discriminator/on_step_real_loss", d_real_loss.mean().item(), num_step)
        writer.add_scalar("Discriminator/on_step_fake_loss", d_fake_loss.mean().item(), num_step)

        d_loss = d_real_loss + d_fake_loss

        d_loss.backward()
        d_optimizer.step()

if __name__ == '__main__':
    if MODE == "train":
        print('Size of training datset: ', trainset_length)
        if not os.path.exists(os.path.join(os.getcwd(), exp_name, 'weights')):
            os.makedirs(os.path.join(os.getcwd(), exp_name, 'weights'))

        print('Training Started!')
        for e in range(NUM_EPOCHS):
            train(e)
            r_scheduler.step(e)
            d_scheduler.step(e)
            if e%2 == 0:
                torch.save(refine_net.state_dict(), os.path.join(os.getcwd(), exp_name, 'weights', "weights_net_epoch_{}.pth".format(e)))
                if e == 0:
                    cmd = f"gsutil -m cp -r {exp_name}/ {save_path}/"
                    os.system(cmd)
                else:
                    try:
                        cmd = f"gsutil -m cp -r {exp_name}/weights/weights_net_epoch_{e}.pth {save_path}/{exp_name}/weights/"
                        os.system(cmd)
                    except:
                        print("Error in storing weights!")
        writer.close()
        cmd = f"gsutil -m cp -r runs/ {save_path}/{exp_name}/"
        os.system(cmd)