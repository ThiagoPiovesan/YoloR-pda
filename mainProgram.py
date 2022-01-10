# /*=========================================================*\
# |      /============================================\       |
# |     ||  - Código para detecção e monitoramento, - ||      |
# |     ||  -     de onças e emissão de alertas     - ||      |
# |     ||  -       Projeto: Jaguar Detector        - ||      |
# |     ||  -   Tecnologia: YOLOR + Object Trackig  - ||      |
# |     ||  -       Módulo: Camera IP + OpenCV      - ||      |
# |     ||  -          Created by WongKinYiu        - ||      |
# |     ||  -Aprimorado por: Thiago P. e Jhonatan R.- ||      |
# |     ||  -          Versao atual: 1.0.0          - ||      |
# |      \============================================/       |
# \*=========================================================*/

# Link do Github: https://github.com/ThiagoPiovesan
# Link do Github: https://github.com/Jhow-Rambo

#==================================================================================================#
# Bibliotecas utilizadas:

import argparse
import os
import platform
import shutil
import time, schedule
from pathlib import Path

import cv2
from numpy.core.records import array
import torch
import torch.backends.cudnn as cudnn
from numpy import random
#--------------------------------------------------------------------------------------------------#
from utils.google_utils import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from models.models import *
from utils.datasets import *
from utils.general import *

#--------------------------------------------------------------------------------------------------#
# Object tracker import

from obj_tracking import CentroidTracker
#--------------------------------------------------------------------------------------------------#
# Bot import

from dotenv import load_dotenv
from threading import Thread
import os

import sys
sys.path.append('bot_jaguar')

import AdminBot
import UserBot

# from bot_jaguar import AdminBot                                         # bot importation
# from bot_jaguar import UserBot                                          # bot importation

#==================================================================================================#
# Instanciando classe Bot Admin

load_dotenv()                                                           # Carregando as chaves | Loading keys
token_admin = os.getenv('ADMIN_TOKEN')                                  # Token do admin       | Admin token
admin_id = os.getenv('ADMIN_ID')                                        # Id do admin          | Admin ID

token_user = os.getenv('USER_TOKEN')                                    # Token do user        | Admin token
user_id = os.getenv('USER_ID')                                          # Id do user           | Admin ID

#bot_admin.send_alert(detection='pessoa', accuracy='92%', img=None)
#==================================================================================================#
# Control Variables 
#TODO: Acrescentar onça

# Bot infos:
class_name: str = "person"                                              # ["person", "jaguar"] | Class to be detected 
accuracy: int = 70                                                      # Acurácia mínima -> Minimum accuracy   

# Log init:
log_active: bool = True                                                 # True -> Log on | False -> Log off.

# Object tracking:
rects = []                                                              # Save the object bounding boxes
ct: object = CentroidTracker()                                          # Instacia do Object tracker

# Send controller:
last_ID: list = [0]                                                    # IDs of the objects in the last frame
send_control: bool = True                                               # Controller to send just 1 fram per time
#==================================================================================================#
# Função para declaração das classes: 

def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)
#==================================================================================================#
# Function to control the actual time and schedule functions:

def checkHour(arg):
    global bot_admin, bot_user
    
    try:
#--------------------------------------------------------------------------------------------------#
        if arg == "alive":
            
            if opt.adminBot:
                bot_admin.send_isAlive()
#--------------------------------------------------------------------------------------------------#
            if opt.userBot:
                bot_user.send_isAlive()
#--------------------------------------------------------------------------------------------------#
        if arg == "change":
            
            if opt.adminBot:
                bot_admin.update_log_date()
#--------------------------------------------------------------------------------------------------#
    except:
        pass
#--------------------------------------------------------------------------------------------------#
# Schedule events:

schedule.every().day.at("08:00").do(checkHour, arg = "alive")
schedule.every().day.at("00:00").do(checkHour, arg = "change")
#==================================================================================================#
# Função principal para detecção dos objetos desejados:

def detect(save_img = False, send_control = True):
    global last_ID, bot_admin, bot_user
    
    prevTime = 0
    out, source, weights, view_img, save_txt, imgsz, cfg, names = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.cfg, opt.names
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
#==================================================================================================#
    if opt.adminBot:
        bot_admin = AdminBot.AdminBot(token_admin, admin_id, log=log_active)    # Instancia do bot do admin

        t1 = Thread(target = bot_admin.alive)
        t1.start()

    if opt.userBot:
        bot_user = UserBot.UserBot(token_user, user_id)    # Instancia do bot do admin
    
#==================================================================================================#
    # Initialize
    device = select_device(opt.device)
    
    if os.path.exists(out):
        shutil.rmtree(out)                                      # delete output folder

    os.makedirs(out)                                            # make new output folder
    half = device.type != 'cpu'                                 # half precision only supported on CUDA

#--------------------------------------------------------------------------------------------------#
    # Load model
    model = Darknet(cfg, imgsz).cuda()                          # if you want cuda remove the comment

    model.load_state_dict(torch.load(weights[0], map_location=device)['model'])
    #model = attempt_load(weights, map_location=device)         # load FP32 model
    #imgsz = check_img_size(imgsz, s=model.stride.max())        # check img_size

    model.to(device).eval()
    if half:
        model.half()                                            # to FP16
#--------------------------------------------------------------------------------------------------#
    # Second-stage classifier
    classify = False                                            # It is optional...

    if classify:
        modelc = load_classifier(name='resnet101', n=2)         # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()
#--------------------------------------------------------------------------------------------------#
    # Set Dataloader
    vid_path, vid_writer = None, None

    if webcam:                                                  # If it is using an webcam
        view_img = True
        cudnn.benchmark = True                                  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)           

    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, auto_size=64)  # Dataset load
#--------------------------------------------------------------------------------------------------#
    # Get names and colors
    names = load_classes(names)                                 # class names load
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
#--------------------------------------------------------------------------------------------------#
    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)      # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()               # uint8 to fp16/32
        img /= 255.0                                            # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
#--------------------------------------------------------------------------------------------------#
        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
#--------------------------------------------------------------------------------------------------#
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()
#--------------------------------------------------------------------------------------------------#
        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
#--------------------------------------------------------------------------------------------------#
        # Process detections
        for i, det in enumerate(pred):                          # detections per image
            
            if webcam:                                          # batch_size >= 1 
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s
#--------------------------------------------------------------------------------------------------#    

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]                       # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]          # normalization gain whwh

#==================================================================================================#
        # Main loop --> Taking frame by frame informations:
            if det is not None and len(det):
#--------------------------------------------------------------------------------------------------#
            # Definição de variáveis auxiliares:         
                ac_array = []
                na_array = []
                rects = []                                      # Clean the bouding boxes
#--------------------------------------------------------------------------------------------------#        
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
#--------------------------------------------------------------------------------------------------#
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()                 # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])        # add to string
#--------------------------------------------------------------------------------------------------#
                # Write results
                for *xyxy, conf, cls in det:
                # Prints to debug:
                    # print('#-------------------------------------------------------------------------------#\n')
                    # print('Conf: ', det)
    
                # Definição do Object tracking:
                    rects.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
                    objects = ct.update(rects)             
#--------------------------------------------------------------------------------------------------#
                # Not using this --> #TODO: Remove thiss...
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
#--------------------------------------------------------------------------------------------------#
                    if save_img or view_img:                    # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)

                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)

                    ac_array.append(float('%.2f' % conf) * 100)
                    na_array.append(names[int(cls)])
#==================================================================================================#      
            # OBJECT TRACKING:
                # loop over the tracked objects
              
                for (objectID, centroid) in objects.items():
                    
                    # draw both the ID of the object and the centroid of the
                    # object on the output frame
                    text = "ID {}".format(objectID)
                    cv2.putText(im0, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (190, 0, 0), 2)
                    cv2.circle(im0, (centroid[0], centroid[1]), 4, (190, 0, 0), -1)
      
#==================================================================================================#
            # SENDING MESSAGE TO THE BOT:       #TODO: Acrescentar para onça também...
                acc = float('%.2f' % conf) * 100 
#==================================================================================================#
            # Copia um instancia da imagem coletada: 
            # Take one snap of the image collected:
                frame = im0.copy()
                # If class_name = person and accuracy >= 80 %
                if (names[int(cls)] == class_name) and (int(acc) >= accuracy):
                    print(last_ID)
                    
                    if (objectID > max(last_ID)) and send_control:
                    # Prints to debug
                        print('\n#------------------------------------------#')
                        print(" Classes: ", na_array, "| Accuracy: ", ac_array, "%")
                        print('#------------------------------------------#\n')
#--------------------------------------------------------------------------------------------------#                    
                    # Save image to computer:
                    
                        cv2.imwrite('capture.png', frame)     # TODO: Encontrar um jeito de mandar o frame
                        photo = open('capture.png', 'rb')
#--------------------------------------------------------------------------------------------------#                 
                    # Sending message to bot:
                        send_control = False
                        
                        #bot_admin.send_alert(detection = names[int(cls)], accuracy = acc, img = photo)
                        bot_admin.send_alert(detection = na_array, accuracy = ac_array, img = photo)
#--------------------------------------------------------------------------------------------------#
                        last_ID.append(objectID)            # Spam messages controller --> Evita de ficar spamando o bot    
#--------------------------------------------------------------------------------------------------#                     
                    else:
                        pass
                else:
                    send_control = True
#==================================================================================================#
            schedule.run_pending()               # Check actual time -> Schedule function    

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))
#--------------------------------------------------------------------------------------------------#
            # Stream results
            if view_img:
                currTime = time.time()
                fps = 1 / (currTime - prevTime)
                prevTime = currTime
#--------------------------------------------------------------------------------------------------#               
                cv2.putText(im0, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 196, 255), 2)
                cv2.imshow(p, im0)
#--------------------------------------------------------------------------------------------------#                
                if cv2.waitKey(30) == 27:                       # q to quit
                    raise StopIteration
#--------------------------------------------------------------------------------------------------#
            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
#--------------------------------------------------------------------------------------------------#
                else:
                    if vid_path != save_path:                   # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()                # release previous video writer

                        fourcc = 'mp4v'                         # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                        
                    vid_writer.write(im0)
#--------------------------------------------------------------------------------------------------#
    if save_txt or save_img:
        print('Results saved to %s' % Path(out))
        if platform == 'darwin' and not opt.update:  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))

#==================================================================================================#
# Função Principal:
# python mainProgram.py --source 0 --weights ../weights/yolor_p6.pt --device 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/yolor_p6.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='YoloR/inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='YoloR/inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=1280, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--cfg', type=str, default='YoloR/cfg/yolor_p6.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='YoloR/data/coco.names', help='*.cfg path')

    parser.add_argument('--adminBot', type=bool, default='True', help='Admin Bot ON/OFF')
    parser.add_argument('--userBot', type=bool, default='False', help='User Bot ON/OFF')
    

    opt = parser.parse_args()
#==================================================================================================#
    print('\n#==================================================================================================#')
    print(opt)
    print('#==================================================================================================#\n')
#==================================================================================================#
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['']:
                detect(send_control)
                strip_optimizer(opt.weights)
        else:
            detect()
#==================================================================================================#
