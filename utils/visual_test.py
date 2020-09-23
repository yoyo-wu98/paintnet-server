import numpy as np
import cv2
import imageio
from matplotlib import pyplot as plt

def plot_imglist(l, result_key=None, right=1.25, divide=6, channels='RGB'):
    '''List the plots of a list
    Input:
        l - list
        result_key - The key of the element whose value we wanna display. In case the element in the list l is a dict
        display arrangement:
            right - spare between images
            divide - how many images listed in a line
            channels - Input images' channels
    '''
    plt.figure(figsize=(10, int(np.ceil(len(l)/divide) * (10 / divide))), dpi=80)
    plt.subplots_adjust(right=right)
    for idx, im in enumerate(l):
        
        if len(l) < divide:
            plt.subplot(1, len(l), idx + 1)
        else:
            plt.subplot(int(np.ceil(len(l)/divide)), divide, idx + 1)
        if channels == 'BGR' or channels == 'BGRA':
            im = cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA)
        if result_key == None:
            plt.imshow(im)
        else:
            plt.imshow(im[result_key])
        plt.xticks([])
        plt.yticks([])
    plt.show()

def show_save_img(img, path=None, channel='RGBA'):
    """
    Default image channel : BGRA
    channel : 'BGRA'/'RGBA'
    """
    if path != None:
        if channel == 'RGB':
            cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGRA))
        elif channel == 'RGBA':
            cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA))
        elif channel == 'BGR' or channel == 'BGRA' or channel == 'GRAY':
            cv2.imwrite(path, img)
    if channel == 'BGRA':
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))
    elif channel == 'BGR':
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGBA))
    elif channel == 'RGB' or channel == 'RGBA':
        plt.imshow(img)
    elif channel == 'GRAY':
        plt.imshow(img, cmap=plt.cm.gray)
    else:
        print("ERROR channel")
        return
    plt.show()


def list_stage_seg(segment_topology_list, img_idx):
    '''List all cropped part in all stages of the topology segmentation
    Print & Plot
    Input:
        segment_topology_list - seg result list
        img_idx - index of the image
    Example:
        list_stage_seg(segment_topology_list, 0)
    '''
    for stage_idx in range(len(segment_topology_list[img_idx])):
        print('stage:', stage_idx)
        print('Seg result:',
              [segment_topology_list[img_idx][stage_idx][seg_idx]['category']
               for seg_idx in list(segment_topology_list[img_idx][stage_idx].keys())])
        plot_imglist([segment_topology_list[img_idx][stage_idx][seg_idx]['cropped']
                      for seg_idx in list(segment_topology_list[img_idx][stage_idx].keys())])

def gif_generate_list(out_list, name, channel='RGBA'):
    '''Generate the process list into a gif
    Input:
        out_list - process list
        name - gif name
        channel - image channel e.g. 'RGB' / 'RGBA'
    '''
    if channel == 'BGR':
        out_list = [cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGBA) for img in out_list]
    if channel == 'BGRA':
        out_list = [cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGRA2RGBA) for img in out_list]
    gif=imageio.mimsave(name+'.gif',[out_list[:][i] for i in range(len(out_list)) if i % 5 == 0],'GIF',duration=0.1)


def video_generate_list(out_list, name, file_path=None, size=None, channel='RGBA'):
    '''Generate the process list into a video
    Input:
        out_list - process list
        name - video name
        channel - image channel e.g. 'RGB' / 'RGBA'
    '''
    if channel == 'BGR' or channel == 'BGRA':
        out_list = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8) for img in out_list]
    if file_path is None:
        file_path = name + ".mp4"
    fps = 60
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    if size is None:
        size = (out_list[0].shape[1], out_list[0].shape[0])
    video = cv2.VideoWriter(file_path, fourcc, fps, size)
    for img in out_list:
        video.write(img)
    video.release()