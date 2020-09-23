# System libs
import os
import time
import argparse
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
# Our libs
from mit_semseg.config import cfg as cfg_ss
from mit_semseg.dataset import TestDataset
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode, find_recursive, setup_logger
from mit_semseg.lib.nn import user_scattered_collate, async_copy_to
from mit_semseg.lib.utils import as_numpy
from PIL import Image
from tqdm import tqdm


def test(segmentation_module, loader, device, names):
    segmentation_module.eval()
    pbar = tqdm(total=len(loader))
    re = []
    for batch_data in loader:
        # process data
        batch_data = batch_data[0]
        segSize = (batch_data['img_ori'].shape[0],
                   batch_data['img_ori'].shape[1])
        img_resized_list = batch_data['img_data']

        with torch.no_grad():
            scores = torch.zeros(1, cfg_ss.DATASET.num_class, segSize[0], segSize[1])
            if torch.cuda.is_available():
                scores = async_copy_to(scores, device)

            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                del feed_dict['img_ori']
                del feed_dict['info']
                if torch.cuda.is_available():
                    feed_dict = async_copy_to(feed_dict, device)

                # forward pass
                pred_tmp = segmentation_module(feed_dict, segSize=segSize)
                scores = scores + pred_tmp / len(cfg_ss.DATASET.imgSizes)

            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())
            pred = np.int32(pred)
            pixs = pred.size

            uniques, counts = np.unique(pred, return_counts=True)
            pred_ratios = {}
            for idx in np.argsort(counts)[::-1]:
                name = names[uniques[idx] + 1]
                ratio = counts[idx] / pixs * 100
                if ratio > 0.1:
                    pred_ratios[name] = ratio
            
            re.append({"original_img" : batch_data['img_ori'],
                     "img_inf" : batch_data['info'],
                      "pred_result" : pred,
                      "pred_ratio" : pred_ratios})

        pbar.update(1)
    return re

def inference_prob(img, device, select_model_option="ade20k-resnet50dilated-ppm_deepsup"): # select_model_option = "ade20k-mobilenetv2dilated-c1_deepsup" / "ade20k-hrnetv2"
    '''Load the data and preprocess settings
    Input:
        img - the path of our target image
        device - Current device running
        select_model_option - name of NN we use
    '''
    cfg_ss.merge_from_file("ss/config/" + select_model_option + ".yaml")

    logger = setup_logger(distributed_rank=0)   # TODO

    cfg_ss.MODEL.arch_encoder = cfg_ss.MODEL.arch_encoder.lower()
    cfg_ss.MODEL.arch_decoder = cfg_ss.MODEL.arch_decoder.lower()

    # absolute paths of model weights
    cfg_ss.MODEL.weights_encoder = os.path.join('ss/' +
        cfg_ss.DIR, 'encoder_' + cfg_ss.TEST.checkpoint)
    cfg_ss.MODEL.weights_decoder = os.path.join('ss/' +
        cfg_ss.DIR, 'decoder_' + cfg_ss.TEST.checkpoint)

    assert os.path.exists(cfg_ss.MODEL.weights_encoder) and os.path.exists(cfg_ss.MODEL.weights_decoder), "checkpoint does not exist!"

    # generate testing image list
    imgs = [img]
    assert len(imgs), "imgs should be a path to image (.jpg) or directory."
    cfg_ss.list_test = [{'fpath_img': x} for x in imgs]

    if not os.path.isdir(cfg_ss.TEST.result):
        os.makedirs(cfg_ss.TEST.result)
    
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg_ss.MODEL.arch_encoder,
        fc_dim=cfg_ss.MODEL.fc_dim,
        weights=cfg_ss.MODEL.weights_encoder)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg_ss.MODEL.arch_decoder,
        fc_dim=cfg_ss.MODEL.fc_dim,
        num_class=cfg_ss.DATASET.num_class,
        weights=cfg_ss.MODEL.weights_decoder,
        use_softmax=True)

    crit = nn.NLLLoss(ignore_index=-1)

    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)

    # Dataset and Loader
    dataset_test = TestDataset(
        cfg_ss.list_test,
        cfg_ss.DATASET)
    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=cfg_ss.TEST.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=5,
        drop_last=True)

    segmentation_module.to(device)

    # Main loop
    return segmentation_module, loader_test