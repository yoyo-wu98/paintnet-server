import numpy as np
from PIL import Image as pilimage

import torch
from torchvision import models
from torchvision.transforms.functional import normalize, resize, to_tensor, to_pil_image
from torchcam.cams import CAM, GradCAM, GradCAMpp, SmoothGradCAMpp, ScoreCAM, SSCAM #, ISSCAM
from torchcam.utils import overlay_mask

from tqdm import tqdm

class cam(object):
    '''Detect and plot the CAM of NN model applying on the target image
    '''
    def __init__(self, model_path='./ss/ckpt/resnet50-19c8e357.pth', gam_type='scoregam'):
        print('wtf')
        self.init_model_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.init_model(model_path=model_path)
        self.init_extractors(gam_type=gam_type)

    def init_model_config(self):
        VGG_CONFIG = {_vgg: dict(input_layer='features', conv_layer='features')
                    for _vgg in models.vgg.__dict__.keys()}

        RESNET_CONFIG = {_resnet: dict(input_layer='conv1', conv_layer='layer4', fc_layer='fc')
                        for _resnet in models.resnet.__dict__.keys()}

        DENSENET_CONFIG = {_densenet: dict(input_layer='features', conv_layer='features', fc_layer='classifier')
                        for _densenet in models.densenet.__dict__.keys()}

        self.MODEL_CONFIG = {
            **VGG_CONFIG, **RESNET_CONFIG, **DENSENET_CONFIG,
            'mobilenet_v2': dict(input_layer='features', conv_layer='features')
        }
    
    def init_model(self, model_path):
        # Pretrained imagenet model
        self.cam_model = models.__dict__['resnet50'](pretrained=False).to(device=self.device)
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        self.cam_model.load_state_dict(torch.load(model_path))
        # cam_model = models.__dict__['resnet50'](pretrained=True).to(device=device)
        self.conv_layer = self.MODEL_CONFIG['resnet50']['conv_layer']
        self.input_layer = self.MODEL_CONFIG['resnet50']['input_layer']
        self.fc_layer = self.MODEL_CONFIG['resnet50']['fc_layer']

    def init_extractors(self, gam_type):
        if gam_type == 'scoregam':
            self.cam_extractors = [ScoreCAM(self.cam_model, self.conv_layer, self.input_layer)]
        elif gam_type == 'gradcampp':
            self.cam_extractors = [GradCAMpp(self.cam_model, self.conv_layer)]
        elif gam_type == 'smoothgradcampp':
            self.cam_extractors = [SmoothGradCAMpp(self.cam_model, self.conv_layer, self.input_layer)]

    def plot_overlays(self, img, category_num=5):
        '''
        Plot the heatmaps of input image.
        Input:
            img - input image
            gam_type - the type of CAM we wanna use:
                            'scoregam' / 'gradcampp' / 'smoothgradcampp'
        Output:
            overlays - list of heatmaps(overlays)
        '''
        pil_img = pilimage.fromarray(img)
        # Preprocess image
        img_tensor = normalize(to_tensor(resize(pil_img, (224, 224))),
                            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).to(device=self.device)
        overlays = []
        for idx, extractor in enumerate(tqdm(self.cam_extractors, leave=False)):
            self.cam_model.zero_grad()
            scores = self.cam_model(img_tensor.unsqueeze(0))
            class_idxs = []
            activation_maps = []
            arr_ = scores.squeeze(0)
            overlay = np.zeros((pil_img.size[1], pil_img.size[0]))
            for _ in tqdm(range(category_num)):
                arr_[arr_.argmax()] = torch.min(scores).item()
                activation_map = extractor(arr_.argmax().item(), scores).cpu()
                heatmap = to_pil_image(activation_map, mode='F')
                overlay += np.asarray(heatmap.resize(pil_img.size, resample=pilimage.BICUBIC))
            overlays.append((overlay - overlay.min()) / (overlay.max() - overlay.min()))
        # Clean data
        extractor.clear_hooks()
        return [np.array(overlay) for overlay in overlays]