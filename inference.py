#!/usr/bin/env python
# coding: utf-8

# ## semantic segmantation
import os
import sys
from distutils.version import LooseVersion

# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
import csv
import cv2
from matplotlib import pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm

# Our libs - graphs
import tarjan

# Our libs - ss
from ss.ss_eval_imgs import *

# Our libs - cam
from matplotlib import cm
import requests
from PIL import Image as pilimage
import torch
from torchvision import models
from torchvision.transforms.functional import normalize, resize, to_tensor, to_pil_image
from torchcam.cams import CAM, GradCAM, GradCAMpp, SmoothGradCAMpp, ScoreCAM, SSCAM #, ISSCAM
from torchcam.utils import overlay_mask

# Our libs - visualization for test
pwd = os.getcwd()
sys.path.append(os.path.abspath(pwd + os.path.sep + "utils"))
from visual_test import plot_imglist, show_save_img, list_stage_seg, gif_generate_list, video_generate_list

# Our libs - km
# #### kmeans - 2 （faster & better）
# we have tried to use some advanced methods to smooth the images 
# so that we can seperate the image into a smooth background and texture, 
# but because of the weak performance of python, we failed. 
# So we turn to the k-means method.
from km_new import Kmeans

# Our libs - Math calculating utilities
from cal_utils import *

# Our libs - Probability distribution calculate
from prob_dist import kmeans_prob_dist, edge_prob_dist

# Our libs - Topology detection
from topology import topology

# Our libs - ge
import imageio

sys.path.append(os.path.abspath(pwd + os.path.sep + "ge" + os.path.sep + "baseline" + os.path.sep + "."))
from DRL.actor import *
from Renderer.stroke_gen import *
from Renderer.model import *



# Inference
class Preprocess_imgs(object):
    '''Preprocessing the target images
    Input:
        img_path - the path of the target image (we have to first save the target image) 
                    its channel should be RGB
    '''
    def __init__(self, img_path, category_info_path='ss/data/object150_info.csv'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.names = {}
        self.names_reversed = {}
        self.load_category_info(category_info_path=category_info_path)

        print('Loading the NN models for ss en/decode')
        seg_model, loader_test = inference_prob(img_path, device=self.device
                                                , select_model_option="ade20k-resnet50dilated-ppm_deepsup")
                                    #            , select_model_option="ade20k-mobilenetv2dilated-c1_deepsup")
        re_prob = test(seg_model, loader_test, self.device, self.names) # cpu - about 45s per image; gpu - about 5s per image
        self.re_prob_padded = self.pad_whole_imglist(re_prob)
        del re_prob
        for i in self.re_prob_padded:
            i.update({"seg_re" : self.img_info_trans_croped(i['original_img'], i['pred_result'], self.names)})
        
        print('Combining all \'seg and kmeans\'d part together')
        for idx, i_re_prob in enumerate(tqdm(self.re_prob_padded)):
            self.re_prob_padded[idx] = self.kmeans_seg_part(i_re_prob, names=self.names)
        
        print('Combining kmeans\'d parts together')
        for idx, img in enumerate(tqdm(self.re_prob_padded)):
            kmeans_re = self.kmeans_combine_img(img)
            img.update({'kmeans': kmeans_re})
            self.re_prob_padded[idx] = img

        print('Calculating the probability distributions based on the kmeans distances and edge distances')
        for idx, i_re_prob in enumerate(tqdm(self.re_prob_padded)):
            self.re_prob_padded[idx] = edge_prob_dist(i_re_prob, per=.25)
            self.re_prob_padded[idx] = kmeans_prob_dist(i_re_prob, per=.25)

        print('Detecting topology relationship')
        self.segment_topology_list = self.seg_topology_list(self.re_prob_padded, names=self.names)

        print('CAM')
        '''Calculate the CAM and overlay it on our target image
        Input:
            target image in unpadded size ([5:-5, 5:-5])
        Output:
            overlay in unpadded size ([5:-5, 5:-5])
        '''
        pbar = tqdm(range(len(self.re_prob_padded)))
        for idx in pbar:
            pbar.set_description("Processing %dth image part" % (idx  + 1))
            tmp_overlays = plot_overlays(self.re_prob_padded[idx]['original_img'][5:-5, 5:-5], gam_type='gradcampp')
            self.re_prob_padded[idx].update({'overlay' : tmp_overlays})

        print('Combine \'Kmeans and edge\'d redefined prob with overlays prob')
        self.prob_list = redefine_prob_list(self.re_prob_padded, edge_nearest_prob=1., edge_farthest_prob=0.0,
                       km_nearest_prob=1., km_farthest_prob=0.0, overlay_alpha=5)

    def load_category_info(self, category_info_path):
        self.names[0] = 'padding' # for strokes in transparent part
        with open(category_info_path) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.names[int(row[0])] = row[5].split(";")[0]
        self.names_reversed = dict(zip(self.names.values(), self.names.keys()))


    def pad_whole_imglist(self, prob_imglist):
        """
        pad the whole image with all padding part follow the 'edge' mode and alpha set to 0
        """
        re = []
        for pb in prob_imglist:
            img_padded = np.pad(pb['original_img'], ((5, 5), (5, 5), (0, 0)), mode='edge')
            img_padded = cv2.cvtColor(img_padded, cv2.COLOR_RGB2RGBA)
            seg_padded = np.pad(pb['pred_result'], ((5, 5), (5, 5)), 
                        mode='constant', constant_values=-1)
            img_padded[:, :, 3][np.where((seg_padded[:, :] == -1))] = 0
            pred_l_padded = list(pb['pred_ratio'].keys())
            pred_l_padded.append('padding')
            re.append({'original_img': img_padded,
                    'img_inf': pb['img_inf'],
                    'pred_result': seg_padded,
                    'pred_l': pred_l_padded})
        return re
    
    def set_transparent_by_seg(self, img, seg, category):
        """Set transparent & Crop by seg
        CV2 - RGBA
        set other category's alpha to 0.
        return img_transparented_croped, (upper, lower, left, right)
        """
        img_test_transparent = img
        img_test_transparent = cv2.cvtColor(img_test_transparent, cv2.COLOR_RGB2RGBA)
        img_test_transparent[:, :, 3][np.where((seg[:, :] != category - 1))] = 0
        available = np.where(seg[:, :] ==  category - 1)
    #     print(available)
        return img_test_transparent[
            available[0].min() - 5 : available[0].max() + 5,
            available[1].min() - 5 : available[1].max() + 5,
        :], (
            available[0].min() - 5, 
            available[0].max() + 5, 
            available[1].min() - 5, 
            available[1].max() + 5
        )

    def img_info_trans_croped(self, img, seg, names, idx_list=None):
        """List seg
        img - target image
        seg - ss result
        names - names dict
        idx_list: main part category's indexs e.g. [0, ...]
        return:
        Dictionary { 
            idx(as in the seg) : {
                category : (name),
                cropped : img,
                xy : (upper, lower, left, right)
            }
        }
        """
        if idx_list is None:
            tmp, counts = np.unique(seg, return_counts=True)
            idx_list = [tmp[i] for i in np.argsort(tmp)[::-1] 
                        if (counts[i] / seg.size * 100) > 0.1]
            if -1 not in idx_list:
                idx_list.append(-1)
        info_d = {}
        for idx in idx_list:
            img_cropped, (upper, lower, left, right) = self.set_transparent_by_seg(img, seg, idx + 1)
            info_d[idx] = {'category': names[idx + 1],
                        'cropped': img_cropped,
                        'xy': (upper, lower, left, right)}
        return info_d

    def kmeans_seg_part(self, i_re_prob, names):
        '''
        Apply K-Means to all segmentsation part of image.
        Input:
            i_re_prob - element in re_prob
            names - names dict
        Return:
            i_re_prob - Updated element
        '''
        pbar = tqdm(list(i_re_prob['seg_re'].keys()), leave=False)
        for i in pbar:
            pbar.set_description("Processing %s's part" % names[i + 1])
            if i == -1:
                continue
            k = Kmeans(k=3, per=.25, display_per=1)
            km_re = k.run(i_re_prob['seg_re'][i]['cropped'])
            i_re_prob['seg_re'][i].update({'kmeans': km_re, 'clusters': [i.centroid for i in k.clusters]})
        return i_re_prob

    def kmeans_combine_img(self, i_prob_padded):
        '''Combine all keams\'d parts together into the whole image canvas
        '''
        tmp_kmeans_canvas = np.zeros(i_prob_padded['original_img'].shape)
        for i_key in i_prob_padded['seg_re']:
            if i_key == -1:
                continue
            xy = i_prob_padded['seg_re'][i_key]['xy']
            tmp_kmeans_part = i_prob_padded['seg_re'][i_key]['kmeans'].copy()
            tmp_kmeans_part[:, :, :][tmp_kmeans_part[:, :, 3]==0] = np.array([0, 0, 0, 0])
            tmp_kmeans_canvas[xy[0] : xy[1],
                            xy[2] : xy[3]] += tmp_kmeans_part
        return tmp_kmeans_canvas.astype(np.int)

    def seg_topology_list(self, re_prob, names):
        '''Detect and list all topology relationships in the target image
        '''
        s_t_list = []
        for i_re_prob in tqdm(re_prob):
            top_tmp = topology(i_re_prob['pred_result'])
            print(top_tmp.display_stage(names))
            s_t_list.append(top_tmp.combine_seg_result(i_re_prob['seg_re']))
        return s_t_list

# ## stroke generation

# 1. 由于非线性函数逼近，DQN本身的收敛就不容易。
# 2. GAN的收敛很困难，这就导致的GAN-DQN的训练极其困难，需要精心的调参以及一点运气。
# 
# 所以我们选择 ddpg

# So we find out that ge's channel is **BGR**

class stroke_generation(object):
    def __init__(self, actor_path='./ge/actor_notrans.pkl'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_path = actor_path
        self.renderer_path = renderer_path
        self.init_models()
    
    def init_models(self):
        self.actor = ResNet(9, 18, 65) # action_bundle = 5, 65 = 5 * 13
        self.actor.load_state_dict(torch.load(self.actor_path))
        self.actor = self.actor.to(self.device).eval()

    def save_img(self, res):
        output = res.detach().cpu().numpy() # d * d, 3, H, W
        output = np.transpose(output, (0, 2, 3, 1))
        output = output[0]
        output = (output * 255).astype('uint8')
        return output

    def decode(self, x, original_canvas, width, wl_dict=None, k=.2):
        original_shape = original_canvas.shape[2:]
        
        x = x.view(-1, 10 + 3)
        tmp_stroke = torch.from_numpy(np.array([1 - draw(i[:10], wl_dict=wl_dict, k=k) for i in x ]))
        tmp_stroke = tmp_stroke.view(-1, width, width, 1)
        tmp_color_stroke = tmp_stroke * x[:, -3:].view(-1, 1, 1, 3)
        stroke = np.zeros((tmp_stroke.shape[0],) + original_shape[:2] + (tmp_stroke.shape[3],))
        color_stroke = np.zeros((tmp_stroke.shape[0],) + original_shape[:2] + (tmp_color_stroke.shape[3],))
        for idx in range(stroke.shape[0]):
            stroke[idx] = np.expand_dims(cv2.resize(tmp_stroke[idx].detach().numpy(), (original_shape[1], original_shape[0])), 2)
            color_stroke[idx] = cv2.resize(tmp_color_stroke[idx].detach().numpy(), (original_shape[1], original_shape[0]))
        stroke = torch.from_numpy(stroke)
        color_stroke = torch.from_numpy(color_stroke)
        stroke = stroke.permute(0, 3, 1, 2)
        color_stroke = color_stroke.permute(0, 3, 1, 2)
        stroke = stroke.view(-1, 5, 1, original_shape[0], original_shape[1])
        color_stroke = color_stroke.view(-1, 5, 3, original_shape[0], original_shape[1])
        res = []
        for i in range(5):
            original_canvas = original_canvas * (1 - stroke[:, i]) + color_stroke[:, i]
            res.append(original_canvas)
        return original_canvas, res

    def paint_part(self, target_img, clusters, max_step=20, init_canvas=None, transparent=True, detail=False):
        '''
        Paint each part we can find in each layer
        Input:
            target_img - target image we want to converge
            clusters - clusters : centroid colours
            max_step - specific part
            ini_canvas - whether refresh
            transparent - whether to set the unrelated part transparent
        '''
        # default variables
        
        width = 128
        
        T = torch.ones([1, 1, width, width], dtype=torch.float32).to(self.device)
        # img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = target_img[:, :, :3] #segment_topology_list[0][1][5]['cropped'][:, :, :3]
        origin_shape = (img.shape[1], img.shape[0])
        
        coord = torch.zeros([1, 2, width, width])
        for i in range(width):
            for j in range(width):
                coord[0, 0, i, j] = i / (width - 1.)
                coord[0, 1, i, j] = j / (width - 1.)
        coord = coord.to(self.device) # Coordconv
        
        if init_canvas is None:
            tmp_canvas = np.array([[np.mean(clusters, axis=0) 
                                    for i in range(width)] for i in range(width)])
            tmp_original_canvas = cv2.cvtColor(cv2.resize(tmp_canvas, 
                                                    (target_img.shape[1], 
                                                    target_img.shape[0])).astype(np.uint8),
                                        cv2.COLOR_RGB2RGBA) # 1 * H * W * 4
        else:
            tmp_canvas = init_canvas.astype(np.uint8)
            tmp_original_canvas = init_canvas.astype(np.uint8).copy() # 1 * H * W * 4
            tmp_canvas = cv2.resize(tmp_canvas, (width, width)).astype(np.uint8)

        original_shape = target_img.shape
        canvas = tmp_canvas[:, :, :3].reshape(1, width, width, -1)
        canvas = np.transpose(canvas, (0, 3, 1, 2))
        canvas = torch.tensor(canvas).to(self.device).float() / 255. # 1 * 3 * width * width
        
        original_canvas = tmp_original_canvas[:, :, :3].reshape(1, tmp_original_canvas.shape[0], tmp_original_canvas.shape[1], -1)
        original_canvas = np.transpose(original_canvas, (0, 3, 1, 2))
        original_canvas = torch.tensor(original_canvas).to(self.device).float() / 255. # 1 * 3 * H * W
        

    #     img = re_prob_padded[0]['original_img'][:, :, :3]
        img = cv2.resize(img, (width, width))
        img = img.reshape(1, width, width, -1) # 1 * width * width * 3
        img = np.transpose(img, (0, 3, 1, 2))
        img = torch.tensor(img).to(self.device).float() / 255.
        
        out_imgs = []
        with torch.no_grad():
            if max_step > 2:
                max_step = max_step // 2
            for i in range(max_step):
    #             print('i', i)
                stepnum = T * i / max_step
    #             print('canvas.shape', canvas.shape)
    #             print('original_canvas.shape', original_canvas.shape)
    #             print('img.shape', img.shape)
    #             print('stepnum', stepnum.shape)
    #             print('coord.shape', coord.shape)
                actions = self.actor(torch.cat([canvas, img, stepnum, coord], 1))
                _, res = decode(actions, original_canvas, width) # TODO: original_canvas seperate canvas to optimize
    #             tmp_original_canvas = np.transpose(tmp_original_canvas, (0, 2, 3, 1)).numpy()[0] # 1 * 3 * H * W ->  Numpy.ndarray (H * W * 3)
                for j in range(5):
                    tmp_img = self.save_img(res[j])
                    tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_RGB2RGBA)
                    if transparent:
                        tmp_img[:, :][np.where((target_img[:, :, 3] == 0))] = np.array([0, 0, 0, 0])
                        original_canvas_tmp = tmp_original_canvas.copy()
                #                     print('original_canvas_tmp.shape', original_canvas_tmp.shape)
                #                     tmp_original_canvas = cv2.cvtColor(original_canvas_tmp, cv2.COLOR_RGB2RGBA)
                        original_canvas_tmp[:, :][np.where((target_img[:, :, 3] != 0))] = np.array([0, 0, 0, 0])
                        tmp_img += original_canvas_tmp
                    out_imgs.append(tmp_img)
                    tmp_original_canvas = tmp_img.copy() # H * W * 4
                    # suojin##########################################################################################
                    #                 print('tmp_original_canvas.shape', tmp_original_canvas.shape)
                    #             print('tmp_canvas.shape', tmp_canvas.shape)
                    original_canvas = tmp_original_canvas[:, :, :3].reshape(1, tmp_original_canvas.shape[0], tmp_original_canvas.shape[1], -1) # numpy.array(1 * H * W * 3)
                    original_canvas = np.transpose(original_canvas, (0, 3, 1, 2))
                    original_canvas = torch.tensor(original_canvas).to(self.device).float() / 255 # Tensor (1 * 3 * H * W) 
                tmp_canvas = cv2.resize(tmp_original_canvas, (width, width))# 1 * width * width * 3
                #             canvas = torch.from_numpy(tmp_canvas).permute(0, 3, 1, 2)
                canvas = tmp_canvas[:, :, :3].reshape(1, width, width, -1)
                canvas = np.transpose(canvas, (0, 3, 1, 2)) # 1 * 3 * width * width
                canvas = torch.tensor(canvas).to(self.device).float() / 255
            
        return out_imgs



    def paint_processes_ge(self, target_img, segment_topology, seg_re, seg_list, 
                        prob_img, filepath, step_rate=1.05, max_iter_per_step=4, per=.25):
        # Video params:
        file_path = filepath
        fps = 60
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        size = (target_img.shape[1], target_img.shape[0])
        video = cv2.VideoWriter(file_path, fourcc, fps, size)
        
        # Zeros image distance
        raw_dis = calc_img_weighted_distance(target_img, prob_img)
        dis_step = 0
        max_prob = np.max(prob_img)
        
        tmp_combine = np.full(target_img.shape, 255, dtype=np.int)
        tmp_combine[:,:, 3] = 0
        
        # Video process
        video_cnt = 0
        video.write(cv2.cvtColor(tmp_combine.copy()[:, :, :3].astype(np.uint8),
                                            cv2.COLOR_BGR2RGB).astype(np.uint8))
        
        
        limit_dis = raw_dis / (step_rate ** dis_step)
        dis_step += 1
        iter_in_step = 0
        
        print('Phrase 1:')
        tmp_canvas_part = tmp_combine.copy()
        while iter_in_step == 0 or( calc_img_weighted_distance(target_img, prob_img, canvas=tmp_canvas_part) >= limit_dis and iter_in_step < max_iter_per_step):
            iter_in_step += 1
            print(iter_in_step, '/', max_iter_per_step, ' : ', calc_img_weighted_distance(target_img, prob_img, canvas=tmp_canvas_part), limit_dis)
            out_imgs = self.paint_part(target_img, 
                                [np.mean(np.mean(target_img, axis=0), axis=0)[:3]], max_step=10, 
                                init_canvas=tmp_canvas_part, 
                                transparent=False
                                )
            for stroke in out_imgs:
                tmp_combine_stroked = tmp_combine.copy()
                tmp_combine_stroked = stroke.copy()
                if video_cnt % 3 == 0:
                    video.write(cv2.cvtColor(tmp_combine_stroked[:, :, :3].astype(np.uint8),
                                            cv2.COLOR_BGR2RGB).astype(np.uint8))
                video_cnt += 1
                tmp_combine = stroke.copy()
                tmp_canvas_part = tmp_combine.copy()
        
        # Detail the whole image:
        print('Phrase 2:')
        
        width = 512
        step_each_width = 20
        max_step = 8
        
        tmp_target = target_img.copy()
        tmp_prob = prob_img.copy()
        tmp_canvas = tmp_combine.copy()
        score_dis = cv2.resize(tmp_prob, (int(tmp_prob.shape[1] * per),
                                            int(tmp_prob.shape[0] * per)))
                
        iter_in_step = 0
        
        tmp_limit_dis = calc_img_weighted_distance(tmp_target, tmp_prob) / (step_rate ** dis_step)
        while iter_in_step == 0 or( calc_img_weighted_distance(tmp_target, tmp_prob,
                                        max_prob, canvas=tmp_canvas) >= tmp_limit_dis and iter_in_step < max_iter_per_step):
            iter_in_step += 1
            print(iter_in_step, '/', max_iter_per_step, ' : ', calc_img_weighted_distance(tmp_target, tmp_prob,
                                                                                        max_prob, canvas=tmp_canvas), tmp_limit_dis)
            
            width = 512
            step_each_width = 20
            max_step = 4
            for _ in tqdm(range(3), leave=False):
                dis_step += 1
                
                list_position = generate_xy_dis_by_prob(tmp_prob, step_each_width)
                prob_bar = tqdm(list_position, leave=False)
                for idx, (x, y) in enumerate(prob_bar):
                    x_l, x_r, y_u, y_l = prob_xy_limit(x, y, width, target_img.shape)
                    tmp_canvas_part, tmp_target_part, tmp_prob_part = crop_part(x_l, x_r, y_u, y_l, 
                                                                                tmp_canvas, tmp_target, tmp_prob=tmp_prob)
    #                 print('tmp_prob', tmp_prob)
    #                 print('tmp_prob_part', tmp_prob_part)
                    
                    
                    tmp_limit_dis_part = calc_img_weighted_distance(tmp_target_part, tmp_prob_part) / (step_rate ** dis_step)
                    if calc_img_weighted_distance(tmp_target_part, tmp_prob_part, max_prob,
                                                        canvas=tmp_canvas_part) > tmp_limit_dis_part:
                        out_imgs = paint_part(tmp_target_part[:, :, :3].astype(np.uint8),
                                            [np.mean(np.mean(target_img, axis=0), 
                                                    axis=0)[:3]],
                                            actor, max_step=max_step,
                                            init_canvas=tmp_canvas_part, transparent=False)
                        if not len(out_imgs): continue
                        for stroke in out_imgs:
                            tmp_combine_stroked = tmp_combine.copy()
                            tmp_combine_stroked[x_l : x_r, y_u : y_l] = stroke.copy()
                            tmp_combine = tmp_combine_stroked.copy()
                            if video_cnt % 3 == 0:
                                video.write(cv2.cvtColor(tmp_combine_stroked[:, :, :3].astype(np.uint8),
                                                    cv2.COLOR_BGR2RGB).astype(np.uint8))
                            video_cnt += 1
                            tmp_combine[x_l : x_r, y_u : y_l] = stroke.copy()
                            tmp_canvas_part = tmp_combine[x_l : x_r, y_u : y_l].copy()
                            tmp_canvas[x_l : x_r, y_u : y_l] = stroke.copy()

                # update parameters
                width = int(width / 2) if width > 32 else 32
                step_each_width = int(step_each_width * 2)
                max_step = int(max_step / 2) if max_step >= 2 else 2
            
        # Transparent details:
        print('Phrase 3: Detail the whole image following the seg topology...')
        sbar = tqdm(range(len(segment_topology)))
        for stage in sbar:
            sbar.set_description("Processing %dth stage " % (stage  + 1))
            original_dis_step = dis_step
            
            for i in tqdm(segment_topology[stage], leave=False):
                if i == -1:
                    continue
                width = 256
                tmp_xy = segment_topology[stage][i]['xy']
                max_step = int(np.ceil(((tmp_xy[1] - tmp_xy[0]) * (tmp_xy[3] - tmp_xy[2])) / (10 * 128 ** 2))) * 1
                step_each_width = int(np.ceil(((tmp_xy[1] - tmp_xy[0]) * (tmp_xy[3] - tmp_xy[2])) / (128 ** 2))) * 2
    #             step_each_width = 20
                max_step = max_step if max_step < 8 else 8
                max_step = max_step if max_step > 2 else 2
                step_each_width = step_each_width if step_each_width < 260 else 260
                
                
                tmp_target = segment_topology[stage][i]['cropped']
                tmp_prob = prob_img[tmp_xy[0] : tmp_xy[1], tmp_xy[2] : tmp_xy[3]].copy()
                tmp_canvas = tmp_combine[tmp_xy[0] : tmp_xy[1], tmp_xy[2] : tmp_xy[3]]
                
                
                iter_in_step = 0
                
                tmp_limit_dis = calc_img_weighted_distance(tmp_target, tmp_prob) / (step_rate ** dis_step)
                while iter_in_step == 0 or( calc_img_weighted_distance(tmp_target, tmp_prob,
                                                max_prob, canvas=tmp_canvas) >= tmp_limit_dis and iter_in_step < max_iter_per_step * 2):
                    iter_in_step += 1
                    print(iter_in_step, '/', max_iter_per_step * 2, ' : ', calc_img_weighted_distance(tmp_target, tmp_prob,
                                                                                        max_prob, canvas=tmp_canvas), tmp_limit_dis)
                    dis_step = original_dis_step
                    for _ in tqdm(range(3), leave=False):
                        dis_step += 1
                        
                        list_position = generate_xy_dis_by_prob(tmp_prob, step_each_width)
                        prob_bar = tqdm(list_position, leave=False)
                        for idx, (x, y) in enumerate(prob_bar):
                            if tmp_target[x, y, 3] == 0: continue
                            x_l, x_r, y_u, y_l = prob_xy_limit(x, y, width, tmp_target.shape)
                            tmp_canvas_part, tmp_target_part, tmp_prob_part = crop_part(x_l, x_r, y_u, y_l, 
                                                                                tmp_canvas, tmp_target, tmp_prob=tmp_prob)
                            
                            tmp_limit_dis_part = calc_img_weighted_distance(tmp_target_part, tmp_prob_part) / (step_rate ** dis_step)
                            if calc_img_weighted_distance(tmp_target_part, tmp_prob_part, max_prob,
                                                            canvas=tmp_canvas_part) > tmp_limit_dis_part:
                                out_imgs = paint_part(tmp_target_part,
                                                    segment_topology[stage][i]['clusters'], max_step=max_step,
                                                    init_canvas=tmp_canvas_part, transparent=False)
                                if not len(out_imgs): continue
                                for stroke in out_imgs:
                                    tmp_combine_stroked = tmp_combine.copy()
                                    tmp_combine_stroked[tmp_xy[0] + x_l : tmp_xy[0] + x_r,
                                                        tmp_xy[2] + y_u : tmp_xy[2] + y_l] = stroke.copy()
                                    tmp_combine = tmp_combine_stroked.copy()
                                    if video_cnt % 3 == 0:
                                        video.write(cv2.cvtColor(tmp_combine_stroked[:, :, :3].astype(np.uint8),
                                                            cv2.COLOR_BGR2RGB).astype(np.uint8))
                                    video_cnt += 1
                                    tmp_combine[tmp_xy[0] + x_l : tmp_xy[0] + x_r,
                                                        tmp_xy[2] + y_u : tmp_xy[2] + y_l] = stroke.copy()
                                    tmp_canvas_part = tmp_combine[tmp_xy[0] + x_l : tmp_xy[0] + x_r, 
                                                                tmp_xy[2] + y_u : tmp_xy[2] + y_l].copy()
                                    tmp_canvas = tmp_combine[tmp_xy[0] : tmp_xy[1], tmp_xy[2] : tmp_xy[3]].copy()
                                    
                    width = int(width / 2) if width > 32 else 32 
        
            dis_step = original_dis_step + 1
        
        width = 128
        ## Repeat !
        print('Phrase 4: Detail along the seg list following the decreasing importance...')
        dis_step = 2
        seg_idx_list = [seg_name_id - 1 for seg_name_id in seg_list]
        if len(seg_idx_list) > 10:
            seg_idx_list = seg_idx_list[:10]
        sbar = tqdm(seg_idx_list)
        for item_idx in sbar:
            sbar.set_description("Processing %s part " % names[item_idx + 1])
            if item_idx == -1:
                continue
            tmp_xy = seg_re[item_idx]['xy']
            max_step = int(np.ceil(((tmp_xy[1] - tmp_xy[0]) * (tmp_xy[3] - tmp_xy[2])) / (10 * 128 ** 2))) * 2
            max_step = max_step if max_step < 16 else 16
            max_step = max_step if max_step > 2 else 2
            step_each_width = int(np.ceil(((tmp_xy[1] - tmp_xy[0]) * (tmp_xy[3] - tmp_xy[2])) / (128 ** 2))) * 4
    #         step_each_width = 20
            step_each_width = step_each_width if step_each_width < 260 else 260
            original_dis_step = dis_step
            
            tmp_target = seg_re[item_idx]['cropped']
            tmp_prob = prob_img[tmp_xy[0] : tmp_xy[1], tmp_xy[2] : tmp_xy[3]].copy()
            tmp_canvas = tmp_combine[tmp_xy[0] : tmp_xy[1], tmp_xy[2] : tmp_xy[3]]
            
            iter_in_step = 0
            tmp_limit_dis = calc_img_weighted_distance(tmp_target, tmp_prob) / (step_rate ** dis_step)
            while iter_in_step == 0 or( calc_img_weighted_distance(tmp_target, tmp_prob, max_prob, canvas=tmp_canvas) > tmp_limit_dis and iter_in_step < max_iter_per_step * 3):
                iter_in_step += 1
                print(iter_in_step, '/', max_iter_per_step * 3, ' : ', calc_img_weighted_distance(tmp_target, tmp_prob,
                                                                                        max_prob, canvas=tmp_canvas), tmp_limit_dis)
                dis_step = original_dis_step
                width = 128
                for _ in tqdm(range(3), leave=False):
                    dis_step += 1
                    
                    list_position = generate_xy_dis_by_prob(tmp_prob, step_each_width)
                    prob_bar = tqdm(list_position, leave=False)
                    for idx, (x, y) in enumerate(prob_bar):
                        if tmp_target[x, y, 3] == 0: continue
                        x_l, x_r, y_u, y_l = prob_xy_limit(x, y, width, tmp_target.shape)
                        tmp_canvas_part, tmp_target_part, tmp_prob_part = crop_part(x_l, x_r, y_u, y_l, 
                                                                            tmp_canvas, tmp_target, tmp_prob=tmp_prob)

                        tmp_limit_dis_part = calc_img_weighted_distance(tmp_target_part, tmp_prob_part) / (step_rate ** dis_step)
                        if calc_img_weighted_distance(tmp_target_part, tmp_prob_part,
                                                    max_prob, canvas=tmp_canvas_part) > tmp_limit_dis_part:

                            out_imgs = paint_part(tmp_target_part,
                                            [np.mean(np.mean(target_img, axis=0), 
                                                    axis=0)[:3]], max_step=max_step,
                                            init_canvas=tmp_canvas_part, transparent=False)
                            if not len(out_imgs): continue
                            for stroke in out_imgs:
                                tmp_combine_stroked = tmp_combine.copy()
                                tmp_combine_stroked[tmp_xy[0] + x_l : tmp_xy[0] + x_r, tmp_xy[2] + y_u : tmp_xy[2] + y_l] = stroke.copy()
                                tmp_combine = tmp_combine_stroked.copy()
                                if video_cnt % 3 == 0:
                                    video.write(cv2.cvtColor(tmp_combine_stroked[:, :, :3].astype(np.uint8),
                                                        cv2.COLOR_BGR2RGB).astype(np.uint8))
                                video_cnt += 1
                                tmp_combine[tmp_xy[0] + x_l : tmp_xy[0] + x_r,
                                            tmp_xy[2] + y_u : tmp_xy[2] + y_l] = stroke.copy()
                                tmp_canvas_part = tmp_combine[tmp_xy[0] + x_l : tmp_xy[0] + x_r,
                                                            tmp_xy[2] + y_u : tmp_xy[2] + y_l].copy()
                                tmp_canvas[x_l : x_r, y_u : y_l] = stroke.copy()
                                
                    width = int(width / 2) if width > 32 else 32
                    
            dis_step = original_dis_step
        
        # Detail the whole image:
        width = 64
        step_each_width = int(np.ceil((target_img.shape[0] * target_img.shape[1]) / (128 ** 2))) * 4
        step_each_width = step_each_width if step_each_width < 260 else 260
        print('Details:')
        for _ in tqdm(range(3)):
            width = int(width / 2) if width > 16 else 16
            step_each_width = int(step_each_width / 2) if step_each_width > 40 else 40
    #         step_each_width = 20
            max_step = int(max_step / 2) if max_step > 2 else 2
            
            
            list_position = generate_xy_dis_by_prob(prob_img, step_each_width)
            prob_bar = tqdm(list_position, leave=False)
            for idx, (x, y) in enumerate(prob_bar):
                x_l, x_r, y_u, y_l = prob_xy_limit(x, y, width, target_img.shape)
                tmp_canvas_part, tmp_target_part, _ = crop_part(x_l, x_r, y_u, y_l, 
                                                                            tmp_combine, target_img)
                out_imgs = paint_part(tmp_target_part,
                                    [np.mean(np.mean(target_img, axis=0), 
                                            axis=0)[:3]], max_step=max_step,
                                    init_canvas=tmp_canvas_part, transparent=False)
                for stroke in out_imgs:
                    tmp_combine_stroked = tmp_combine.copy()
                    tmp_combine_stroked[x_l : x_r, y_u : y_l] = stroke.copy()
                    tmp_combine = tmp_combine_stroked.copy()
                    if video_cnt % 3 == 0:
                        video.write(cv2.cvtColor(tmp_combine_stroked[:, :, :3].astype(np.uint8),
                                            cv2.COLOR_BGR2RGB).astype(np.uint8))
                    video_cnt += 1
        video.release()
        return tmp_combine

# tmp_combines = []
# for idx in range(len(re_prob_padded)):
#     tmp_combine = paint_processes_ge(re_prob_padded[idx]['original_img'],
#                        segment_topology_list[idx],
#                         re_prob_padded[idx]['seg_re'],
#                        [names_reversed[name] for name in re_prob_padded[idx]['pred_l']],
#                        prob_list[idx],
#                        actor,
#                         filepath='out'+str(idx)+'.mp4', max_iter_per_step=1)
#     tmp_combines.append(tmp_combine)

# # gif_generate_list(processes, name='md', channel='BGRA')
# gif_generate_list([processes[idx][:, :, :3].astype(np.uint8) for idx in range(len(processes)) if idx % 3 == 0],
#                   name='out')#, channel='BGRA')

# # gif_generate_list(processes, name='md', channel='BGRA')
# video_generate_list([processes[idx][:, :, :3].astype(np.uint8) for idx in range(len(processes)) if idx % 3 == 0],
#                   name='out', channel='BGRA')



