# ***************************************************************************************
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.                    *
#                                                                                       *
# Permission is hereby granted, free of charge, to any person obtaining a copy of this  *
# software and associated documentation files (the "Software"), to deal in the Software *
# without restriction, including without limitation the rights to use, copy, modify,    *
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to    *
# permit persons to whom the Software is furnished to do so.                            *
#                                                                                       *
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,   *
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A         *
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT    *
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION     *
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE        *
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                                *
# ***************************************************************************************
import argparse
import os
import shutil
import warnings

import cv2
import gluoncv
import matplotlib.image as mpimg
import mxnet as mx
import requests
from gluoncv.data.transforms.presets.segmentation import test_transform
from gluoncv.utils.viz import get_color_pallete
from mxnet import image
from mxnet.gluon.data.vision import transforms


def video2frame(video_src_path, video, frames_save_path, frame_width, frame_height, interval):
    """
    Extract frame from video by interval
    :param video_src_path: video src path
    :param video:　video file name
    :param frames_save_path:　save path
    :param frame_width:　frame widty
    :param frame_height:　frame height
    :param interval:　interval for frame to extract
    :return:　frame images
    """
    video_name = video[:-4].split('/')[-1]
    print ("reading video ：", video_name)

    os.makedirs(frames_save_path + video_name, exist_ok=True)
    each_frame_save_full_path = os.path.join(frames_save_path, video_name) + "/"
    print('each_frame_save_full_path ' + each_frame_save_full_path)
    each_video_full_path = os.path.join(video_src_path, video)
    print('each_video_full_path is ' + each_video_full_path)
    cap = cv2.VideoCapture(each_video_full_path)
    frame_index = 0
    frame_count = 0
    if cap.isOpened():
        success = True
    else:
        success = False
        print("Read failed!")

    while(success):
        success, frame = cap.read()

        if frame_index % interval == 0:
            print("---> Reading the %d frame:" % frame_index, success)
            resize_frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
            # cv2.imwrite(each_video_save_full_path + each_video_name + "_%d.jpg" % frame_index, resize_frame)
            cv2.imwrite(each_frame_save_full_path + "%d.png" % frame_count, resize_frame,[int( cv2.IMWRITE_JPEG_QUALITY), 95])
            frame_count += 1

        frame_index += 1
 
    cap.release()
    return each_frame_save_full_path


if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_server_address', type=str, default='')
    parser.add_argument('--video_formats', type=list, default=[".mp4", ".avi"])
    parser.add_argument('--image_width', type=int, default=1280)
    parser.add_argument('--image_height', type=int, default=720)
    parser.add_argument('--frame_time_interval', type=int, default=1000)
    
    args, _ = parser.parse_known_args()
    
    print('Received arguments {}'.format(args))

    # Use GPU if available
    if len(mx.test_utils.list_gpus())!=0:#return range(0, 4)
        ctx=mx.gpu() #default gpu(0)
    else:
        ctx=mx.cpu()
    
    model = gluoncv.model_zoo.get_model('deeplab_resnet101_citys', pretrained=True, ctx=ctx) # load the pretrained model trained on Cityscapes dataset
    
    input_data_path = '/opt/ml/processing/input_data'
    output_data_path = '/opt/ml/processing/output_data'
    print('Reading input data from {}'.format(input_data_path))
    
    
    # Mock call API within VPC, change this link and logic accordingly 
    api_server_addr = args.api_server_address
    r=requests.get("http://" + api_server_addr)
    print(r.text)
    
    # Extract frame from videos
    videos_src_path = input_data_path
    frames_save_path = "/opt/ml/processing/frame_data/"
    video_formats = args.video_formats
    width = args.image_width
    height = args.image_height
    time_interval = args.frame_time_interval
    
    video_dirs = os.listdir(input_data_path)

    for video in video_dirs:
        if video.endswith(('.mp4', '.avi')):
            frame_save_full_path = video2frame(input_data_path, video, frames_save_path, width, height, time_interval)
            print('frame_save_full_path is ' + frame_save_full_path)
            
            file_dirs = os.listdir(frame_save_full_path)
            video_name = video[:-4].split('/')[-1]

            # inference by using pretrained model
            for file in file_dirs:
                if file.endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                    img = image.imread(os.path.join(frame_save_full_path, file))
                    img = test_transform(img, ctx)
                    output = model.predict(img)
                    predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()
                    mask = get_color_pallete(predict, 'citys')
                    mask.save(os.path.join(output_data_path, video_name + '_' + file))

        else:
            continue
        
