import time
import numpy as np
import os
import cv2
import torch
from numpy import random
from torchvision import transforms
from yolo.models.experimental import attempt_load
from yolo.utils.general import non_max_suppression, scale_coords, check_img_size
from yolo.utils.torch_utils import  TracedModel
from yolo.weight.rec.model import CNN

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class YoloPredict:
    def __init__(self):
        self.imgsz = 416
        self.conf_thres = 0.5
        self.iou_thres = 0.45
        self.convert_tensor = transforms.ToTensor()
        self.model_crop, self.model_seg, self.model_rec, self.model_defaut = self.init_model()

    def init_model(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        weights_crop = ['yolo/weight/crop/best.pt']
        try:
            model_crop = attempt_load(weights_crop, map_location=self.device)
        
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                torch.cuda.empty_cache()
                model_crop = attempt_load(weights_crop, map_location=self.device)
            
        self.stride_crop = int(model_crop.stride.max())
        
        model_crop = TracedModel(model_crop, self.device, self.imgsz)
        model_crop.half()
        model_crop(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(model_crop.parameters())))
        # model_crop.eval()
        print("第一個模型成功")

        weights_seg = ['yolo/weight/seg/best.pt']
        model_seg = attempt_load(weights_seg, map_location=self.device) 

        self.stride_seg = int(model_seg.stride.max())
        model_seg = TracedModel(model_seg, self.device, self.imgsz)
        model_seg.half()
        model_seg(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(model_seg.parameters())))
        model_seg.eval()
        print("第二個模型成功")

        self.LABEL_DICT = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'P', 25: 'Q', 26: 'R', 27: 'S', 28: 'T', 29: 'U', 30: 'V', 31: 'W', 32: 'X', 33: 'Y', 34: 'Z', 35: '2', 36: '3', 37: '6', 38: '9', 39: 'B', 40: 'K', 41: 'R'}
        model_rec = CNN()
        model_rec.load_state_dict(torch.load("yolo/weight/rec/model.pt"))
        model_rec.eval()
        print("第三個模型成功")

        weights_seg = ['yolo/weight/default/yolov7.pt']
        model_defaut = attempt_load(weights_seg, map_location=self.device)
        model_defaut = TracedModel(model_defaut, self.device, 640)
        model_defaut.half()
        model_defaut(torch.zeros(1, 3, 640, 640).to(self.device).type_as(next(model_defaut.parameters())))
        model_defaut.eval()
        print("第四個模型成功")

        self.def_stride = int(model_defaut.stride.max())
        self.def_imgsz = check_img_size(640, s=self.def_stride)
        self.def_names = model_defaut.module.names if hasattr(model_defaut, 'module') else model_defaut.names
        self.def_colors = [[random.randint(200, 255) for _ in range(3)] for _ in self.def_names]
        self.stride_model_defaut = int(model_defaut.stride.max())
        return model_crop, model_seg, model_rec, model_defaut

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)

    def predict_result(self, model, stride, origin_img, imgsz):
        img = self.letterbox(origin_img, imgsz, stride=stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half()
        img /= 255.0
        img = img.unsqueeze(0)

        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=True)[0]
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=None, agnostic=False)[0]
        pred = [pred[pred[:, 0].sort(descending=True)[1]]]

        output = {
            "box": list(),
            "image": list(),
            "label": list(),
            "color": list(),
        }
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], origin_img.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                    output["box"].append([c1, c2])
                    output["image"].append(origin_img[c1[1]:c2[1], c1[0]:c2[0]])
                    output["label"].append(f'{self.def_names[int(cls)]} {conf:.2f}')
                    output["color"].append(self.def_colors[int(cls)])
        return output

    def predict_yolo(self, model, stride, origin_img):
        # cv2.imshow('My Image', origin_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        img = self.letterbox(origin_img, self.imgsz, stride=stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half()
        img /= 255.0
        img = img.unsqueeze(0)
        
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=True)[0]
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=None, agnostic=False)[0]
        pred = [pred[pred[:, 0].sort(descending=True)[1]]]
        
        output = {
            "box":list(),
            "image":list(),
        }

        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], origin_img.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                    output["box"].append([c1, c2])
                    output["image"].append(origin_img[c1[1]:c2[1], c1[0]:c2[0]])
                    
        return output

    def predict_rec(self, img):
        img = cv2.resize(img, (32, 32))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV) 
        img_tensor = self.convert_tensor(img)
        img_tensor = torch.unsqueeze(img_tensor, dim=1)
        predic_label, predict_img = self.model_rec(img_tensor)
        predict = torch.max(predic_label, 1)[1].item()
        # print("predict : " + str(LABEL_DICT[predict]))
        return str(self.LABEL_DICT[predict])

    def predict(self, origin_img):
        # origin_img = cv2.imread(img_path)
        start = time.time()

        # crop the car plate license
        crop_output = self.predict_yolo(self.model_crop, self.stride_crop, origin_img)
        # crop_output = self.predict_result(self.model_crop, self.stride_crop, origin_img, self.imgsz)
        # print("crop done !!!")
        if crop_output["box"] != []:
            labels = list()
            # segment the car plate license
            for index, crop_img in enumerate(crop_output["image"]):
                if index > 6:
                    break
                seg_output = self.predict_yolo(self.model_seg, self.stride_seg, crop_img)

                # predict the word
                rec_output = ""
                for seg_img in seg_output["image"]:
                    rec_output += self.predict_rec(seg_img)
                labels.append(rec_output)

            for crop_box, label in zip(crop_output["box"], labels):
                x1, y1, x2, y2 = crop_box[0][0], crop_box[0][1], crop_box[1][0], crop_box[1][1]
                
                fontFace = cv2.FONT_HERSHEY_COMPLEX
                fontScale = 1e-3 * origin_img.shape[0]
                thickness = 1
                labelSize = cv2.getTextSize(label, fontFace, fontScale, thickness)
                _x2 = x1 + labelSize[0][0]  # topright x of text
                _y2 = y1 - labelSize[0][1]  # topright y of text
                cv2.rectangle(origin_img, (x1, y1), (x2, y2), (200, 255, 255), 1)
                
                cv2.rectangle(origin_img, (x1, y1), (_x2, _y2), (200, 255, 255), cv2.FILLED)
                if len(label) > 8:
                    label = label[:9]
                cv2.putText(origin_img,
                            "{}".format(label),
                            (x1, y1),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale,
                            (0, 0, 0),
                            thickness)

        return origin_img

    def predict_default(self, origin_img):

        start = time.time()
        def_output = self.predict_result(self.model_defaut, self.stride_model_defaut, origin_img, 640)
        # predict the word
        try:
            # print("predict process time : " + str(time.time() - start))
            for crop_box, label, color in zip(def_output["box"], def_output["label"], def_output["color"]):
                # print("Car Plate License : " + label)
                x1, y1, x2, y2 = crop_box[0][0], crop_box[0][1], crop_box[1][0], crop_box[1][1]
                fontFace = cv2.FONT_HERSHEY_COMPLEX
                fontScale = 1e-3 * origin_img.shape[0]
                thickness = 1
                labelSize = cv2.getTextSize(label, fontFace, fontScale, 2)
                _x2 = x1 + labelSize[0][0]  # topright x of text
                _y2 = y1 - labelSize[0][1]  # topright y of text
                cv2.rectangle(origin_img, (x1, y1), (x2, y2), color, 1)
                
                cv2.rectangle(origin_img, (x1, y1), (_x2, _y2), color, cv2.FILLED)
                if len(label) > 8:
                    label = label[:9]
                cv2.putText(origin_img,
                            "{}".format(label),
                            (x1, y1),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale,
                            (0, 0, 0),
                            thickness)
        except Exception as e:
            print(e)
        return origin_img
