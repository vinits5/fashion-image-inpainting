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

        "cloth_mask": tf.io.FixedLenFeature([], tf.string),
        "cloth": tf.io.FixedLenFeature([], tf.string),

        "gt_warped_mask": tf.io.FixedLenFeature([], tf.string),
        "gt_warped_cloth": tf.io.FixedLenFeature([], tf.string),

        "full_mask": tf.io.FixedLenFeature([], tf.string),
        "occluded_cloth_mask": tf.io.FixedLenFeature([], tf.string),

        "person": tf.io.FixedLenFeature([], tf.string),
        "personMask": tf.io.FixedLenFeature([], tf.string),

        'tps_warped_cloth' :  tf.io.FixedLenFeature([], tf.string),
        'tps_warped_mask':  tf.io.FixedLenFeature([], tf.string),
        'inpaint_region':  tf.io.FixedLenFeature([], tf.string),

        'eroded_tps_warped_cloth':  tf.io.FixedLenFeature([], tf.string),
        'eroded_tps_warped_mask':  tf.io.FixedLenFeature([], tf.string),
        'dialated_inpaint_region':  tf.io.FixedLenFeature([], tf.string),
    }

    parsed_features = tf.io.parse_single_example(proto, keys_to_features)

    cloth_no = parsed_features["clothno"]
    person_no = parsed_features["personno"]
    
    cloth = (tf.cast(tf.image.decode_jpeg(parsed_features["cloth"], channels=3), tf.float32) / 255.0 - 0.5) / 0.5
    cloth_mask = tf.cast(tf.image.decode_png(parsed_features["cloth_mask"], channels=1), tf.float32) / 255.0

    gt_warped_mask = tf.cast(tf.image.decode_png(parsed_features["gt_warped_mask"], channels=1), tf.float32) / 255.0
    gt_warped_cloth = (tf.cast(tf.image.decode_jpeg(parsed_features["gt_warped_cloth"], channels=3), tf.float32) / 255.0 - 0.5) / 0.5

    full_mask = tf.cast(tf.image.decode_png(parsed_features["full_mask"], channels=1), tf.float32) / 255.0
    occluded_cloth_mask = tf.cast(tf.image.decode_png(parsed_features["occluded_cloth_mask"], channels=1), tf.float32) / 255.0
    
    person = (tf.cast(tf.image.decode_jpeg(parsed_features["person"], channels=3), tf.float32) / 255.0 - 0.5) / 0.5
    grapy = tf.cast(tf.image.decode_png(parsed_features["personMask"], channels=1), tf.float32)

    tps_warped_cloth = (tf.cast(tf.image.decode_png(parsed_features["tps_warped_cloth"], channels=1), tf.float32) / 255.0 - 0.5) / 0.5
    tps_warped_mask = tf.cast(tf.image.decode_png(parsed_features["tps_warped_mask"], channels=1), tf.float32) / 255.0
    inpaint_region = tf.cast(tf.image.decode_png(parsed_features["inpaint_region"], channels=1), tf.float32) / 255.0

    eroded_tps_warped_cloth = (tf.cast(tf.image.decode_png(parsed_features["eroded_tps_warped_cloth"], channels=1), tf.float32) / 255.0 - 0.5) / 0.5
    eroded_tps_warped_mask = tf.cast(tf.image.decode_png(parsed_features["eroded_tps_warped_mask"], channels=1), tf.float32) / 255.0
    dialated_inpaint_region = tf.cast(tf.image.decode_png(parsed_features["dialated_inpaint_region"], channels=1), tf.float32) / 255.0
    
    cloth_mask = tf.cast(cloth_mask > 0, tf.float32)
    p_sil = tf.cast(grapy > 0, tf.float32)
    person = person * p_sil + (1 - p_sil)

    # Multiplying by cloth mask 
    cloth = cloth * cloth_mask + (1 - cloth_mask)

    grid_img = tf.cast(
        tf.image.decode_png(tf.io.read_file("gs://experiments_logs/datasets/gradient_2.png")), tf.float32
    )
    grid_img = (grid_img / 255.0 - [0.5, 0.5, 0.5]) / [0.5, 0.5, 0.5]

    person_cloth = person*occluded_cloth_mask
    warped_cloth_input = person_cloth - person_cloth*inpaint_region

    data = {
        "cloth": cloth,
        "cloth_mask": cloth_mask,
        "person": person,

        "gt_warped_mask": gt_warped_mask,
        "gt_warped_cloth": gt_warped_cloth,

        "grid_img": grid_img,
        "occluded_cloth_mask": occluded_cloth_mask,

        "personno": person_no,
        "clothno": cloth_no,
        "person_cloth": person_cloth,

        "inpaint_region": inpaint_region,
        "warped_cloth": warped_cloth_input

    }

    model_inputs = {
        "cloth": cloth,
        "occluded_cloth_mask": occluded_cloth_mask,
        "inpaint_region": inpaint_region,
        "warped_cloth": warped_cloth_input
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

    # dataset = dataset.filter(filter_func)

    if mode != "k_worst":
        num_lines = sum(1 for _ in dataset)
        # num_lines = 15000
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