import argparse
import time
from pathlib import Path
import numpy as np

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental_simple import attempt_load

from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, \
        apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box, plot_center
from utils.datasets import letterbox
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

class Detector:
    def __init__(self, weights, img_size, device):
        with torch.no_grad():
            self.weights = weights

            set_logging()

            # Initialize Device and Precision
            print(torch.cuda.is_available())
            self.device = select_device(device)
            self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

            # Load model
            self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
            self.stride = int(self.model.stride.max())  # model stride
            self.img_size = check_img_size(img_size, s=self.stride)  # check img_size
            trace=True
            if trace:
                self.model = TracedModel(self.model, self.device, self.img_size)
            if self.half:
                self.model.half()  # to FP16
            # Get names and colors
            self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
            self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

    def detect(self, img0, conf, show_detection):
        with torch.no_grad():
            # Run inference
            if self.device.type != 'cpu':
                self.model(torch.zeros(1, 3, self.img_size, self.img_size).to(self.device).type_as(next(self.model.parameters())))  # run once
            old_img_w = old_img_h = self.img_size
            old_img_b = 1

            # Padded resize
            img = letterbox(img0, self.img_size, stride=self.stride)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

            # Convert to tesor
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            pred = self.model(img, augment=False)[0]

            # Apply NMS
            pred = non_max_suppression(pred, conf, 0.45, classes=None, agnostic=False)
            dets = []
            for i, det in enumerate(pred):
                if len(det):
                    det_new = np.zeros((det.shape[0], det.shape[1]+2), dtype=float)
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                    det_new[:, :4] = det.cpu().numpy()[:, :4]
                    det_new[:, 4] = det_new[:, 0] + (det_new[:, 2] - det_new[:, 0]) / 2
                    det_new[:, 5] = det_new[:, 1] + (det_new[:, 3] - det_new[:, 1]) / 2
                    det_new[:, 6:] = det.cpu().numpy()[:, 4:]
                    if show_detection:
                        for *xyxy,cx, cy, conf, cls in reversed(det_new):
                            label = f'{self.names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, img0, label=label, color=self.colors[int(cls)], line_thickness=1)
                            plot_center(cx, cy, img0, color=self.colors[int(cls)])
                        #dets.append({'class':cls.item(), 'label':self.names[int(cls)], 'conf':round(conf.item(), 2),
                        #    'x_min':xyxy[0], 'y_min':xyxy[1], 'x_max':xyxy[2], 'y_max':xyxy[3]})
                    dets.append(det_new)
            if show_detection:
                # Show the AI view, rescale to output size
                img0 = cv2.resize(img0, (int(img0.shape[1]/2),int(img0.shape[0]/2)))
                cv2.imshow("AI vision", img0)
                cv2.waitKey(1)
            # Returns a list of detections in format [x_min, y_min, x_max, y_max, conf, class]
            try:
                return np.array(dets)[0]
            except:
                print("Warning did not detect anything!")
                return None


