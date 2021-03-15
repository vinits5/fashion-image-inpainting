from torchvision import transforms
from torch.utils import data

from utils import CentralErasing

from PIL import Image

import os
import re
import csv
import h5py
import glob
import shutil
import random
import numpy as np
import torch
import tensorflow as tf

def filterMask(mask, valList):
    newmask = tf.cast(mask == valList[0], tf.float32)
    for val in valList[1:]:
        newmask += tf.cast(mask == val, tf.float32)
    return newmask

def _parse(proto):
    keys_to_features = {
        "clothno": tf.io.FixedLenFeature([], tf.int64),
        "personno": tf.io.FixedLenFeature([], tf.int64),
        "clothMask": tf.io.FixedLenFeature([], tf.string),
        "person": tf.io.FixedLenFeature([], tf.string),
        "cloth": tf.io.FixedLenFeature([], tf.string),
        "personMask": tf.io.FixedLenFeature([], tf.string),
        "full": tf.io.FixedLenFeature([], tf.string),
        "densepose": tf.io.FixedLenFeature([], tf.string),
        "torso_gt": tf.io.FixedLenFeature([], tf.string),

        "lua": tf.io.FixedLenFeature([], tf.string),
        "rua": tf.io.FixedLenFeature([], tf.string),
        "lla": tf.io.FixedLenFeature([], tf.string),
        "rla": tf.io.FixedLenFeature([], tf.string),

        "cloth_collar_orientation": tf.io.FixedLenFeature([], tf.string),
        "person_collar_orientation": tf.io.FixedLenFeature([], tf.string),
        "person_collar_segmentation": tf.io.FixedLenFeature([], tf.string),
        "cloth_collar_segmentation": tf.io.FixedLenFeature([], tf.string),
        "collar_present": tf.io.FixedLenFeature([], tf.string),
        "shoulder_left": tf.io.FixedLenFeature([], tf.string),
        "shoulder_right": tf.io.FixedLenFeature([], tf.string),
        "sleeves": tf.io.FixedLenFeature([], tf.string),
    }

    parsed_features = tf.io.parse_single_example(proto, keys_to_features)
    cloth = (tf.cast(tf.image.decode_jpeg(parsed_features["cloth"], channels=3), tf.float32) / 255.0 - 0.5) / 0.5
    cloth_mask = tf.cast(tf.image.decode_png(parsed_features["clothMask"], channels=1), tf.float32)
    person = (tf.cast(tf.image.decode_jpeg(parsed_features["person"], channels=3), tf.float32) / 255.0 - 0.5) / 0.5
    grapy = tf.cast(tf.image.decode_png(parsed_features["personMask"], channels=1), tf.float32)
    torso_gt = tf.cast(tf.image.decode_png(parsed_features["torso_gt"], channels=1), tf.float32)

    lua = tf.cast(tf.image.decode_png(parsed_features["lua"], channels=1), tf.float32)
    rua = tf.cast(tf.image.decode_png(parsed_features["rua"], channels=1), tf.float32)
    lla = tf.cast(tf.image.decode_png(parsed_features["lla"], channels=1), tf.float32)
    rla = tf.cast(tf.image.decode_png(parsed_features["rla"], channels=1), tf.float32)

    sleeves = parsed_features["sleeves"]
    densepose = tf.image.decode_png(parsed_features["densepose"], channels=3)
    dp_seg = tf.cast(densepose[..., 0], tf.int32)
    dp_seg = tf.cast(tf.one_hot(dp_seg, depth=25), tf.float32)
    dp_uv = (tf.cast(densepose[..., 1:], tf.float32) / 255.0 - 0.5) / 0.5
    person_no = parsed_features["personno"]
    cloth_no = parsed_features["clothno"]

    full_mask = lua + rua + lla + rla + torso_gt
    full_mask = tf.cast(full_mask > 0, tf.float32)
    
    cloth_mask = tf.cast(cloth_mask > 0, tf.float32)
    p_sil = tf.cast(grapy > 0, tf.float32)
    person = person * p_sil + (1 - p_sil)

    # Multiplying by cloth mask 
    cloth = cloth * cloth_mask + (1 - cloth_mask)
    gt_warped_mask = full_mask
    gt_warped_cloth = person * gt_warped_mask
    occluded_cloth_mask = filterMask(grapy, [5, 6, 7])
    grid_img = tf.cast(
        tf.image.decode_png(tf.io.read_file("gs://experiments_logs/datasets/gradient_2.png")), tf.float32
    )
    grid_img = (grid_img / 255.0 - [0.5, 0.5, 0.5]) / [0.5, 0.5, 0.5]

    person_cloth = person*occluded_cloth_mask

    data = {
        "cloth": cloth,
        "cloth_mask": cloth_mask,
        "person": person,
        "gt_warped_mask": gt_warped_mask,
        "gt_warped_cloth": gt_warped_cloth,
        "grid_img": grid_img,
        "sleeves": sleeves,
        "occluded_cloth_mask": occluded_cloth_mask,
        "personno": person_no,
        "clothno": cloth_no,
        "person_cloth": person_cloth
    }

    model_inputs = {
        "cloth": cloth,
        "occluded_cloth_mask": occluded_cloth_mask
    }
    return data, model_inputs

def _filter(data, model_inputs):
    string = data["sleeves"]
    return tf.strings.regex_full_match(string, "short-sleeves")

def create_dataset(parse_func, filter_func, tfrecord_path, num_data, batch_size, mode, data_split, device):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    if mode == "train":
        dataset = dataset.take(int(data_split * num_data))
        dataset = dataset.shuffle(2048, reshuffle_each_iteration=True)

    elif mode == "val":
        dataset = dataset.skip(int(data_split * num_data))

    elif mode == "k_worst":
        dataset = dataset.take(data_split * num_data)

    dataset = dataset.map(
        parse_func,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    dataset = dataset.filter(filter_func)

    if mode != "k_worst":
        # num_lines = sum(1 for _ in dataset)
        num_lines = 15000
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size, drop_remainder=True)
    else:
        num_lines = num_data  # doesn't get used anywhere
        dataset = dataset.batch(batch_size, drop_remainder=False)

    if device != "colab_tpu":
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset, num_lines
    else:
        return dataset

def define_dataset(tfrecord_path, batch_size, train=True):
    per_replica_train_batch_size = batch_size
    per_replica_val_batch_size = batch_size
    if train:
        data_gen, dataset_length = create_dataset(
            parse_func=_parse,
            filter_func=_filter,
            tfrecord_path=tfrecord_path,
            num_data=37129,
            batch_size=per_replica_train_batch_size,
            mode="train",
            data_split=0.8,
            device='gpu',
        )
    else:
        data_gen, dataset_length = create_dataset(
            parse_func=_parse,
            filter_func=_filter,
            tfrecord_path=tfrecord_path,
            num_data=37129,
            batch_size=per_replica_val_batch_size,
            mode="val",
            data_split=0.8,
            device='gpu',
        )
    return data_gen, dataset_length