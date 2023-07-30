from PIL import  Image
import tempfile
import cv2
from super_gradients.training import models
import torch
import numpy as np
import math
import streamlit as st
import time
import easyocr
from ultralytics import YOLO
from sort import *

reader = easyocr.Reader(['en'], gpu=False)

def ocr_image(ttt, x1, y1, x2, y2):
    ttt = ttt[y1:y2, x1:x2]

    gray = cv2.cvtColor(ttt, cv2.COLOR_BGR2GRAY)

    result = reader.readtext(gray)

    print(result)
    text = ""

    for res in result:
        if len(result) == 1:
            text = res[1]
        if len(result) > 1 and len(res[1]) > 6 and res[2] > 0.2:
            text = res[1]

    return str(text)

def load_yolonas_process_each_image(image, confidence, st):

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    model = models.get('yolo_nas_s', pretrained_weights = "coco").to(device)

    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"
                  ]

    result = list(model.predict(image, conf=confidence))[0]
    bbox_xyxys = result.prediction.bboxes_xyxy.tolist()
    confidences = result.prediction.confidence
    labels = result.prediction.labels.tolist()

    for (bbox_xyxy, confidence, cls) in zip(bbox_xyxys, confidences, labels):
        bbox = np.array(bbox_xyxy)
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        x1, y1, x2, y2  = int(x1), int(y1), int(x2), int(y2)
        classname = int(cls)
        class_name = classNames[classname]
        conf = math.ceil((confidence*100))/100
        label = f'{class_name}{conf*100}'+"%"
        t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
        c2 = x1 + t_size[0], y1 - t_size[1] - 3
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255),3)
        cv2.rectangle(image, (x1, y1), c2, [255, 144, 30], -1, cv2.LINE_AA)
        cv2.putText(image, label, (x1, y1-2), 0, 1, [255, 255,255], thickness=1, lineType = cv2.LINE_AA)
    st.subheader('Output Image')
    st.markdown('사람 외 여러가지 객체 식별입니다.')
    st.markdown('사람 감지 시, 빛 또는 소리 신호와 연계할 수 있습니다.')
    st.markdown('사람 감지 시, 지게차 등 장비의 엔진 제어와 연계하여 충돌을 방지하는 안전 장치를 만들 수 있습니다.')
    st.image(image, channels = 'BGR', use_column_width=True)


def load_yolonas_process_cntr_image(image, confidence, st):


    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model = YOLO('yolov8n.pt')
    model = YOLO('C:/python/udemy/YOLO_NAS_StreamLit_Course/runs/detect/train/weights/best.pt')
    classNames = ["container-no", "size", "tare"]

    result = list(model.predict(image, conf=confidence))[0]
    bbox_xyxys = result.boxes.xyxy.tolist()
    confidences = result.boxes.conf
    labels = result.names

    for i, (bbox_xyxy, confidence, cls) in enumerate(zip(bbox_xyxys, confidences, labels)):
        if i >= 3:
            break
        bbox = np.array(bbox_xyxy)
        # print(bbox)
        if len(bbox) >= 4:  # Ensure bbox has at least 4 values
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            classname = int(cls)
            class_name = classNames[classname]
            conf = math.ceil((confidence * 100)) / 100
            # Call ocr_image inside the loop for each object
            # text = ocr_image(image, x1, y1, x2, y2)
            # label = text
            label = f'{class_name}{int(conf*100)}'+"%"
            # label = f'{class_name}{conf}' + "  OCR : " + text
            # print(text)
            # print("Frame N", count, "", x1, y1, x2, y2)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 1)
            t_size = cv2.getTextSize(label, 0, fontScale=0.1, thickness=1)[0]
            c2 = x1-10, y1 - t_size[1]+10
            cv2.rectangle(image, (x1-(x2-x1)-100, y1 + 25), c2, [255, 144, 30], -1, cv2.LINE_AA)
            cv2.putText(image, label, (x1-(x2-x1)-100, y1 + 25), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
    # resize_frame = cv2.resize(image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)


    st.subheader('Output Image')
    st.markdown('특정 영역만 사전 학습하여 검출 영역의 글자만 추출하는 시도입니다.')
    st.markdown('OCR 정확도를 높이는 고민중입니다.')
    st.markdown('TEST 결과 OCR은 실시간 CAM에서 버퍼링이 발생 했고 이미지에서의 검출이 안정적이었습니다.')
    st.image(image, channels = 'BGR', use_column_width=True)

    bbox_xyxys2 = result.boxes.xyxy.tolist()

    for i, bbox_xyxy in enumerate(bbox_xyxys2):
        if i >= 3:
            break
        bbox = np.array(bbox_xyxy)
        if len(bbox) >= 4:  # Ensure bbox has at least 4 values
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Call ocr_image inside the loop for each object
            text = ocr_image(image, x1, y1, x2, y2)
            st.subheader(text)
#
# def load_yolonas_process_count_frame(video_name, kpi1_text, kpi2_text, kpi3_text, stframe):
#     cap = cv2.VideoCapture(video_name)
#
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
#     frame_width = int(cap.get(3))
#
#     frame_height = int(cap.get(4))
#
#     device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
#
#     model = models.get('yolo_nas_m', pretrained_weights="coco").to(device)
#
#     count = 0
#     prev_time = 0
#     classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
#                   "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
#                   "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
#                   "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
#                   "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
#                   "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
#                   "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
#                   "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
#                   "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
#                   "teddy bear", "hair drier", "toothbrush"
#                   ]
#     out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
#     st.subheader('연구 방향')
#     st.markdown('식별 후 Count 연계입니다.')
#     st.markdown('물류 현장에서는 파레트 등 포장재 재고 조사시 활용해 볼 수 있습니다.')
#     tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
#     totalcountup = []
#     totalcountdown = []
#     limitdown = [225, 850, 963, 850]
#     limitup = [979, 850, 1667, 850]
#     while True:
#         ret, frame = cap.read()
#         count += 1
#         if ret:
#             detections = np.empty((0, 5))
#             result = list(model.predict(frame, conf=0.35))[0]
#             bbox_xyxys = result.prediction.bboxes_xyxy.tolist()
#             confidences = result.prediction.confidence
#             labels = result.prediction.labels.tolist()
#             for (bbox_xyxy, confidence, cls) in zip(bbox_xyxys, confidences, labels):
#                 bbox = np.array(bbox_xyxy)
#                 x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
#                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#                 classname = int(cls)
#                 class_name = classNames[classname]
#                 conf = math.ceil((confidence * 100)) / 100
#                 currentArray = np.array([x1, y1, x2, y2, conf])
#                 detections = np.vstack((detections, currentArray))
#             resultsTracker = tracker.update(detections)
#             cv2.line(frame, (limitup[0], limitup[1]), (limitup[2], limitup[3]), (255, 0, 0), 5)
#             cv2.line(frame, (limitdown[0], limitdown[1]), (limitdown[2], limitdown[3]), (255, 0, 0), 5)
#             for result in resultsTracker:
#                 x1, y1, x2, y2, id = result
#                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#                 cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
#                 cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 144, 30), 3)
#                 label = f'{int(id)}'
#                 t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
#                 c2 = x1 + t_size[0], y1 - t_size[1] - 3
#                 cv2.rectangle(frame, (x1, y1), c2, [255, 144, 30], -1, cv2.LINE_AA)
#                 cv2.putText(frame, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
#                 if limitup[0] < cx < limitup[2] and limitup[1] - 15 < cy < limitup[3] + 15:
#                     if totalcountup.count(id) == 0:
#                         totalcountup.append(id)
#                         cv2.line(frame, (limitup[0], limitup[1]), (limitup[2], limitup[3]), (0, 255, 0), 5)
#                 if limitdown[0] < cx < limitdown[2] and limitdown[1] - 15 < cy < limitdown[3] + 15:
#                     if totalcountdown.count(id) == 0:
#                         totalcountdown.append(id)
#                         cv2.line(frame, (limitdown[0], limitdown[1]), (limitdown[2], limitdown[3]), (0, 255, 0), 5)
#             # cv2.putText(frame, str("Vehicle Entering") + ":" + str(len(totalcountup)), (1317, 91), cv2.FONT_HERSHEY_PLAIN, 0, 1, (255, 255,255), 2)
#             # cv2.putText(frame, str("Vehicle Leaving") + ":" + str(len(totalcountdown)), (141, 91), cv2.FONT_HERSHEY_PLAIN, 0, 1, (255, 255, 255), 2)
#             cv2.rectangle(frame, (1267, 65), (1617, 97), [255, 0, 255], -1, cv2.LINE_AA)
#             cv2.putText(frame, str("Vehicle Entering") + ":" + str(len(totalcountup)), (1317, 91), 0, 1,
#                         [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
#             cv2.rectangle(frame, (100, 65), (441, 97), [255, 0, 255], -1, cv2.LINE_AA)
#             cv2.putText(frame, str("Vehicle Leaving") + ":" + str(len(totalcountdown)), (141, 91), 0, 1,
#                         [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
#             #resize_frame = cv2.resize(frame, (0, 0), fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
#             current_time = time.time()
#             fps = 1/(current_time - prev_time)
#             prev_time = current_time
#             #out.write(frame)
#             #cv2.imshow("Frame", resize_frame)
#             #if cv2.waitKey(1) & 0xFF == ord('1'):
#                 #break
#             yield frame, int(len(totalcountup)), int(len(totalcountdown)), fps
#
#             kpi1_text.write(f"<h1 style='text-align:center; color:red;'>{'{:.1f}'.str(int(len(totalcountup)))}</h1>", unsafe_allow_html=True)
#             kpi2_text.write(f"<h1 style='text-align:center; color:red;'>{'{:.1f}'.str(int(len(totalcountdown)))}</h1>", unsafe_allow_html=True)
#             kpi1_text.write(f"<h1 style='text-align:center; color:red;'>{'{:.1f}'.format(fps)}</h1>", unsafe_allow_html=True)
#
#         else:
#             break
#


def load_yolonas_process_empty_frame(video_name, kpi1_text, kpi2_text, kpi3_text, stframe):
    cap = cv2.VideoCapture(video_name)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_width = int(cap.get(3))

    frame_height = int(cap.get(4))

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    model = models.get('yolo_nas_s', num_classes = 1, checkpoint_path = 'C:/python/udemy/YOLO_NAS_StreamLit_Course/empty/ckpt_best.pth')

    count = 0
    prev_time = 0
    classNames = ["Empty Shelf"]

    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
    st.subheader('연구 방향')
    st.markdown('빈공간 식별 입니다.')
    st.markdown('물류 현장에서는 실시간 영상 기반 선반 Rack 또는 평치Zone의 보관 Capacity를 분석에 활용할 수 있습니다.')
    while True:
        ret, frame = cap.read()
        count += 1
        if ret:
            result = list(model.predict(frame, conf=0.35))[0]
            bbox_xyxys = result.prediction.bboxes_xyxy.tolist()
            confidences = result.prediction.confidence
            labels = result.prediction.labels.tolist()
            for (bbox_xyxy, confidence, cls) in zip(bbox_xyxys, confidences, labels):
                bbox = np.array(bbox_xyxy)
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                classname = int(cls)
                class_name = classNames[classname]
                conf = math.ceil((confidence * 100)) / 100
                label = f'{class_name}{conf}'
                # print("Frame N", count, "", x1, y1, x2, y2)
                # # t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                # c2 = x1 + t_size[0], y1 - t_size[1] - 3
                # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                # cv2.rectangle(frame, (x1, y1), c2, [255, 144, 30], -1, cv2.LINE_AA)
                # cv2.putText(frame, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
                overlay = frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)  # Draw the transparent red area
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 1)
                alpha = 0.5  # Transparency factor (0.0 to 1.0)
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            resize_frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8, interpolation=cv2.INTER_AREA)

            stframe.image(frame, channels='BGR', use_column_width = True)
            current_time = time.time()

            fps = 1/(current_time - prev_time)

            prev_time = current_time

            kpi1_text.write(f"<h1 style='text-align:center; color:red;'>{'{:.1f}'.format(fps)}</h1>", unsafe_allow_html=True)
            kpi2_text.write(f"<h1 style='text-align:center; color:red;'>{'{:.1f}'.format(width)}</h1>", unsafe_allow_html=True)
            kpi3_text.write(f"<h1 style='text-align:center; color:red;'>{'{:.1f}'.format(height)}</h1>", unsafe_allow_html=True)

        else:
            break

def load_yolo_nas_process_each_frame(video_name, kpi1_text, kpi2_text, kpi3_text, stframe):
    cap = cv2.VideoCapture(video_name)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_width = int(cap.get(3))

    frame_height = int(cap.get(4))

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    model = models.get('yolo_nas_s', pretrained_weights="coco").to(device)

    count = 0
    prev_time = 0
    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"
                  ]

    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
    st.subheader('연구 방향')
    st.markdown('사람 감지 시, 빛 또는 소리 신호와 연계할 수 있습니다.')
    st.markdown('사람 감지 시, 지게차 등 장비의 엔진 제어와 연계하여 충돌을 방지하는 안전 장치를 만들 수 있습니다.')
    while True:
        ret, frame = cap.read()
        count += 1
        if ret:
            result = list(model.predict(frame, conf=0.65))[0]
            bbox_xyxys = result.prediction.bboxes_xyxy.tolist()
            confidences = result.prediction.confidence
            labels = result.prediction.labels.tolist()
            for (bbox_xyxy, confidence, cls) in zip(bbox_xyxys, confidences, labels):
                bbox = np.array(bbox_xyxy)
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                classname = int(cls)
                class_name = classNames[classname]
                conf = math.ceil((confidence * 100)) / 100
                label = f'{class_name}{conf}'
                print("Frame N", count, "", x1, y1, x2, y2)
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                cv2.rectangle(frame, (x1, y1), c2, [255, 144, 30], -1, cv2.LINE_AA)
                cv2.putText(frame, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
            stframe.image(frame, channels='BGR', use_column_width = True)
            current_time = time.time()

            fps = 1/(current_time - prev_time)

            prev_time = current_time

            kpi1_text.write(f"<h1 style='text-align:center; color:red;'>{'{:.1f}'.format(fps)}</h1>", unsafe_allow_html=True)
            kpi2_text.write(f"<h1 style='text-align:center; color:red;'>{'{:.1f}'.format(width)}</h1>", unsafe_allow_html=True)
            kpi3_text.write(f"<h1 style='text-align:center; color:red;'>{'{:.1f}'.format(height)}</h1>", unsafe_allow_html=True)

        else:
            break



def load_yolonas_process_safe_frame(video_name, kpi1_text, kpi2_text, kpi3_text, stframe):
    cap = cv2.VideoCapture(video_name)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_width = int(cap.get(3))

    frame_height = int(cap.get(4))

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    model = models.get('yolo_nas_s', num_classes=7, checkpoint_path= 'C:/python/udemy/YOLO_NAS_StreamLit_Course/safe/ckpt_best.pth').to(device)

    count = 0
    prev_time = 0
    classNames = ['Protective Helmet', 'Shield', 'Jacket', 'Dust Mask', 'Eye Wear', 'Glove', 'Protective Boots']

    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
    st.subheader('설명')
    st.markdown('Protective Helmet, Shield, Jacket, Dust Mask, Eye Wear, Glove, Protective Boots 식별 입니다.')
    st.markdown('물류 현장에서는 실시간 영상 기반 감지 시, 빛 또는 소리 신호와 연계할 수 있습니다')
    while True:
        ret, frame = cap.read()
        count += 1
        if ret:
            result = list(model.predict(frame, conf=0.35))[0]
            bbox_xyxys = result.prediction.bboxes_xyxy.tolist()
            confidences = result.prediction.confidence
            labels = result.prediction.labels.tolist()
            for (bbox_xyxy, confidence, cls) in zip(bbox_xyxys, confidences, labels):
                bbox = np.array(bbox_xyxy)
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                classname = int(cls)
                class_name = classNames[classname]
                conf = math.ceil((confidence * 100)) / 100
                label = f'{class_name}{conf}'
                print("Frame N", count, "", x1, y1,x2, y2)
                t_size = cv2.getTextSize(label, 0, fontScale = 1, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] -3
                cv2.rectangle(frame, (x1, y1), c2, [255, 144, 30], -1, cv2.LINE_AA)
                cv2.putText(frame, label, (x1, y1-2), 0, 1, [255, 255, 255], thickness=1, lineType = cv2.LINE_AA)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)

            stframe.image(frame, channels='BGR', use_column_width = True)
            current_time = time.time()

            fps = 1/(current_time - prev_time)

            prev_time = current_time

            kpi1_text.write(f"<h1 style='text-align:center; color:red;'>{'{:.1f}'.format(fps)}</h1>", unsafe_allow_html=True)
            kpi2_text.write(f"<h1 style='text-align:center; color:red;'>{'{:.1f}'.format(width)}</h1>", unsafe_allow_html=True)
            kpi3_text.write(f"<h1 style='text-align:center; color:red;'>{'{:.1f}'.format(height)}</h1>", unsafe_allow_html=True)

        else:
            break



def main():
    st.title("물류 AI 연구소")
    st.sidebar.title("Settings")
    st.sidebar.subheader("Parameters")
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 300px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 300px;
            margin-left: -300px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    app_mode = st.sidebar.selectbox('Mode 선택', ['App 소개', '현장 안전(Imege)', '현장 안전(Video/cam)', '컨테이너 OCR',
                                    '빈공간 식별', '보호장구'])

    if app_mode == 'App 소개':
        st.markdown('이 프로젝트는 Image, Video, Web-cam 객체 검출 + OCR + 생성형AI 에 관한 연구입니다. ')
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
                width: 300px;
            }
            [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
                width: 300px;
                margin-left: -300px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.video('C:/python/udemy/YOLO_NAS_StreamLit_Course/cntr.mp4')
        st.markdown('''
                    # About Me \n
                    안녕하세요. 이정훈입니다.
                    - [E-mail] ljhun0424@gmail.com
                    ''')

    elif app_mode == '현장 안전(Imege)':
        st.sidebar.markdown('---')
        confidence = st.sidebar.slider('정확도', min_value=0.0, max_value=1.0)
        st.sidebar.markdown('---')
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
                width: 300px;
            }
            [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
                width: 300px;
                margin-left: -300px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        img_file_buffer = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

        DEMO_IMAGE = 'C:/python/udemy/YOLO_NAS_StreamLit_Course/safety/safety-workplan.jpg'

        if img_file_buffer is not None:
            img = cv2.imdecode(np.fromstring(img_file_buffer.read(), np.uint8), 1)
            image = np.array(Image.open(img_file_buffer))
        else:
            img = cv2.imread(DEMO_IMAGE)
            image = np.array(Image.open(DEMO_IMAGE))
        st.sidebar.text('Orignal Image')
        st.sidebar.image(image)
        load_yolonas_process_each_image(img, confidence, st)

    elif app_mode == '현장 안전(Video/cam)':
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
                width: 300px;
            }
            [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
                width: 300px;
                margin-left: -300px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.sidebar.markdown('---')
        use_webcam = st.sidebar.checkbox('Use Webcam')
        st.sidebar.markdown('---')
        video_file_buffer = st.sidebar.file_uploader("Upload a Video", type = ["mp4", "avi", "mov", "asf"])

        DEMO_VIDEO = 'C:/python/udemy/YOLO_NAS_StreamLit_Course/safety/safety_video.mp4'
        # DEMO_VIDEO = 'Video/bikes.mp4'
        # st.sidebar.text('Orignal Video')
        tffile = tempfile.NamedTemporaryFile(suffix= '.mp4', delete=False)

        if not video_file_buffer:
            if use_webcam:
                tffile.name = 0
            else:
                vid = cv2.VideoCapture(DEMO_VIDEO)
                tffile.name = DEMO_VIDEO
                demo_vid = open(tffile.name, 'rb')
                demo_bytes = demo_vid.read()
                st.sidebar.text('Input Video')
                st.sidebar.video(demo_bytes)
        else:
            tffile.write(video_file_buffer.read())
            demo_vid = open(tffile.name, 'rb')
            demo_bytes = demo_vid.read()
            st.sidebar.text('Input Video')
            st.sidebar.video(demo_bytes)
        stframe = st.empty()
        st.markdown("<hr/>", unsafe_allow_html = True)
        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1:
            st.markdown("**Frame Rate**")
            kpi1_text = st.markdown("0")
        with kpi2:
            st.markdown("**Width**")
            kpi2_text = st.markdown("0")
        with kpi3:
            st.markdown("**Height**")
            kpi3_text = st.markdown("0")
        st.markdown("<hr/>", unsafe_allow_html = True)
        load_yolo_nas_process_each_frame(tffile.name, kpi1_text,kpi2_text,  kpi3_text, stframe)

    elif app_mode == '컨테이너 OCR':
        st.sidebar.markdown('---')
        confidence = st.sidebar.slider('정확도', min_value=0.0, max_value=1.0)
        st.sidebar.markdown('---')
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
                width: 300px;
            }
            [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
                width: 300px;
                margin-left: -300px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        img_file_buffer = st.sidebar.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"])

        DEMO_IMAGE = 'C:/python/udemy/YOLO_NAS_StreamLit_Course/cimage2.jpg'

        if img_file_buffer is not None:
            img = cv2.imdecode(np.fromstring(img_file_buffer.read(), np.uint8), 1)
            image = np.array(Image.open(img_file_buffer))
        else:
            img = cv2.imread(DEMO_IMAGE)
            # img = cv2.resize(img,  (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            image = np.array(Image.open(DEMO_IMAGE))
        st.sidebar.text('원본 이미지')
        st.sidebar.image(image)
        load_yolonas_process_cntr_image(img, confidence, st)
    #
    # elif app_mode == '부자재 재고 count':
    #     st.markdown(
    #         """
    #         <style>
    #         [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
    #             width: 300px;
    #         }
    #         [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
    #             width: 300px;
    #             margin-left: -300px;
    #         }
    #         </style>
    #         """,
    #         unsafe_allow_html=True,
    #     )
    #     st.sidebar.markdown('---')
    #     use_webcam = st.sidebar.checkbox('Use Webcam')
    #     st.sidebar.markdown('---')
    #     video_file_buffer = st.sidebar.file_uploader("Upload a Video", type = ["mp4", "avi", "mov", "asf"])
    #
    #     DEMO_VIDEO = 'C:/python/udemy/YOLO_NAS_StreamLit_Course/Video/video1.mp4'
    #     # st.sidebar.text('Orignal Video')
    #     tffile = tempfile.NamedTemporaryFile(suffix= '.mp4', delete=False)
    #
    #     if not video_file_buffer:
    #         if use_webcam:
    #             tffile.name = 0
    #         else:
    #             vid = cv2.VideoCapture(DEMO_VIDEO)
    #             tffile.name = DEMO_VIDEO
    #             demo_vid = open(tffile.name, 'rb')
    #             demo_bytes = demo_vid.read()
    #             st.sidebar.text('Input Video')
    #             st.sidebar.video(demo_bytes)
    #     else:
    #         tffile.write(video_file_buffer.read())
    #         demo_vid = open(tffile.name, 'rb')
    #         demo_bytes = demo_vid.read()
    #         st.sidebar.text('Input Video')
    #         st.sidebar.video(demo_bytes)
    #     stframe = st.empty()
    #     st.markdown("<hr/>", unsafe_allow_html = True)
    #     kpi1, kpi2, kpi3 = st.columns(3)
    #     with kpi1:
    #         st.markdown("**Frame Rate**")
    #         kpi1_text = st.markdown("0")
    #     with kpi2:
    #         st.markdown("**Width**")
    #         kpi2_text = st.markdown("0")
    #     with kpi3:
    #         st.markdown("**Height**")
    #         kpi3_text = st.markdown("0")
    #     st.markdown("<hr/>", unsafe_allow_html = True)
    #     load_yolonas_process_count_frame(tffile.name, kpi1_text, kpi2_text, kpi3_text, stframe)

    elif app_mode == '빈공간 식별':
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
                width: 300px;
            }
            [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
                width: 300px;
                margin-left: -300px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.sidebar.markdown('---')
        use_webcam = st.sidebar.checkbox('Use Webcam')
        st.sidebar.markdown('---')
        video_file_buffer = st.sidebar.file_uploader("Upload a Video", type = ["mp4", "avi", "mov", "asf"])

        DEMO_VIDEO = 'C:/python/udemy/YOLO_NAS_StreamLit_Course/empty/demo2.mp4'
        # st.sidebar.text('Orignal Video')
        tffile = tempfile.NamedTemporaryFile(suffix= '.mp4', delete=False)

        if not video_file_buffer:
            if use_webcam:
                tffile.name = 0
            else:
                vid = cv2.VideoCapture(DEMO_VIDEO)
                tffile.name = DEMO_VIDEO
                demo_vid = open(tffile.name, 'rb')
                demo_bytes = demo_vid.read()
                st.sidebar.text('Input Video')
                st.sidebar.video(demo_bytes)
        else:
            tffile.write(video_file_buffer.read())
            demo_vid = open(tffile.name, 'rb')
            demo_bytes = demo_vid.read()
            st.sidebar.text('Input Video')
            st.sidebar.video(demo_bytes)
        stframe = st.empty()
        st.markdown("<hr/>", unsafe_allow_html = True)
        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1:
            st.markdown("**Frame Rate**")
            kpi1_text = st.markdown("0")
        with kpi2:
            st.markdown("**Width**")
            kpi2_text = st.markdown("0")
        with kpi3:
            st.markdown("**Height**")
            kpi3_text = st.markdown("0")
        st.markdown("<hr/>", unsafe_allow_html = True)
        load_yolonas_process_empty_frame(tffile.name, kpi1_text, kpi2_text, kpi3_text, stframe)


    elif app_mode == '보호장구':
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
                width: 300px;
            }
            [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
                width: 300px;
                margin-left: -300px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.sidebar.markdown('---')
        use_webcam = st.sidebar.checkbox('Use Webcam')
        st.sidebar.markdown('---')
        video_file_buffer = st.sidebar.file_uploader("Upload a Video", type = ["mp4", "avi", "mov", "asf"])

        DEMO_VIDEO = 'C:/python/udemy/YOLO_NAS_StreamLit_Course/Video/demo3.mp4'
        # st.sidebar.text('Orignal Video')
        tffile = tempfile.NamedTemporaryFile(suffix= '.mp4', delete=False)

        if not video_file_buffer:
            if use_webcam:
                tffile.name = 0
            else:
                vid = cv2.VideoCapture(DEMO_VIDEO)
                tffile.name = DEMO_VIDEO
                demo_vid = open(tffile.name, 'rb')
                demo_bytes = demo_vid.read()
                st.sidebar.text('Input Video')
                st.sidebar.video(demo_bytes)
        else:
            tffile.write(video_file_buffer.read())
            demo_vid = open(tffile.name, 'rb')
            demo_bytes = demo_vid.read()
            st.sidebar.text('Input Video')
            st.sidebar.video(demo_bytes)
        stframe = st.empty()
        st.markdown("<hr/>", unsafe_allow_html = True)
        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1:
            st.markdown("**Frame Rate**")
            kpi1_text = st.markdown("0")
        with kpi2:
            st.markdown("**Width**")
            kpi2_text = st.markdown("0")
        with kpi3:
            st.markdown("**Height**")
            kpi3_text = st.markdown("0")
        st.markdown("<hr/>", unsafe_allow_html = True)
        load_yolonas_process_safe_frame(tffile.name, kpi1_text, kpi2_text, kpi3_text, stframe)

if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
