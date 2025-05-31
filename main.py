

from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import numpy as np
from SORT import Sort 
import os


#  Function to calculate Intersection over Union (IoU) 
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


cap =  cv2.VideoCapture(0)
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Traffic light config 
light_duration = {'green': 41, 'yellow': 4, 'red': 8}
light_colors = ['green', 'yellow', 'red']
light_bgr = {'green': (0, 255, 0), 'yellow': (0, 255, 255), 'red': (0, 0, 255)}
frame_counter = 0
light_index = 0
last_switch_time = time.time()

### CRUCIAL PART OF THE CODE where you need to set the coordinates based on your video pixel resolution
forbid_vehicle_zone = np.array([[345,570],[680,645],[860,580],[530,520]],dtype=np.int32)
forbid_ped_zone = np.array([[0,665],[0,685],[555,685],[860,580],[555,550]],dtype=np.int32)



#Dowmload and load the model in your dir
model = YOLO('yolov8x.pt') 
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]


prev_frame_time = 0
track_id_to_class_id = {}    # Dictionary to store class ID for each track ID
last_known_violation_state = {}      # {track_id: is_violator_boolean}
img_dir = './violater_img'


if not os.path.exists(img_dir):
    os.makedirs(img_dir)
    print(f'Directory "{img_dir}" created successfully')


while True:
    new_frame_time = time.time()
    fps = 1/(new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
    prev_frame_time = new_frame_time

    success, img= cap.read()
    if not success:
        break

    img_display = cv2.resize(img, (1280, 720)) #Resizing the image
    img_for_detection = img_display.copy() # Using a clean copy for detection

   #Traffic Light Simulation Logic
    current_light = light_colors[light_index]
    bar_x, bar_y, bar_width, circle_radius, spacing, padding= 50, 50, 50, 15, 10, 10
    housing_x1 = bar_x - circle_radius - padding
    housing_y1 = bar_y - circle_radius - padding
    last_light_center_y = bar_y + 2*(2*circle_radius + spacing)
    housing_x2 = bar_x + circle_radius + padding
    housing_y2 = last_light_center_y + circle_radius + padding

    current_time = time.time()
    elapsed_time = current_time - last_switch_time
    if elapsed_time >= light_duration[current_light]:
        light_index = (light_index + 1) % 3
        current_light = light_colors[light_index]
        last_switch_time = current_time
    cv2.rectangle(img_display,(housing_x1, housing_y1),(housing_x2, housing_y2),(0, 0, 0),2)     
    for idx, color_name in enumerate(light_colors):
        center_y = bar_y + idx * (2 * circle_radius + spacing)
        color_bgr_val = light_bgr[color_name] if color_name == current_light else (50, 50, 50)
        cv2.circle(img_display, (bar_x, center_y), circle_radius, color_bgr_val, -1)
        
    if current_light in ['green', 'yellow']: 
      cv2.polylines(img_display, [forbid_ped_zone], True, (0, 0, 255), 2)
    else: 
       cv2.polylines(img_display, [forbid_vehicle_zone], True, (0, 0, 255), 2)
    

    #Feeding images to YOLO for detection
    results = model(img_for_detection, stream=True, verbose=False) 
    detections_for_sort = np.empty((0, 5))
    current_frm_detection_with_cls = [] # List of (x1,y1,x2,y2, class_id) to compare with tracker IDs

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = math.ceil((box.conf[0] * 100)) / 100
            # This is the class ID for THIS specific YOLO detection
            original_cls_id = int(box.cls[0])
            if classNames[original_cls_id] in ['car', 'truck', 'bus', 'motorbike','bicycle', 'person'] and conf > 0.5:
                #This part below stacks the detections for tracking using Sort
                detections_for_sort = np.vstack((detections_for_sort, np.array([x1, y1, x2, y2, conf])))
                # Store the bounding box and its class ID
                current_frm_detection_with_cls.append( (x1, y1, x2, y2, original_cls_id) )

    resultsTracker = tracker.update(detections_for_sort)
    vehicles = ['car', 'motorbike', 'bus', 'truck', 'bicycle']
    for track_result in resultsTracker:
        tx1, ty1, tx2, ty2, track_id = track_result
        tx1, ty1, tx2, ty2 = int(tx1), int(ty1), int(tx2), int(ty2)
        w, h = tx2 - tx1, ty2 - ty1



        #Since Sort is a used for tracking algorithm and does not define class label
        # We will use IOU to find the best match
        
        current_object_class_id = -1  #Default class id

        if track_id in track_id_to_class_id:
            current_object_class_id = track_id_to_class_id[track_id]
        else:
            best_iou = 0
            best_match_class_id = -1
            tracked_bbox = (tx1, ty1, tx2, ty2)

            for yolo_x1, yolo_y1, yolo_x2, yolo_y2, yolo_cls_id in current_frm_detection_with_cls:
                yolo_bbox = (yolo_x1, yolo_y1, yolo_x2, yolo_y2)
                iou = calculate_iou(tracked_bbox, yolo_bbox)
                
                # If this YOLO detection is a good match for the tracked box
                if iou > 0.5 and iou > best_iou: 
                    best_iou = iou
                    best_match_class_id = yolo_cls_id
            
            if best_match_class_id != -1:
                track_id_to_class_id[track_id] = best_match_class_id
                current_object_class_id = best_match_class_id

        #Using current_object_class_id for labeling
     
        if current_object_class_id != -1 and current_object_class_id < len(classNames):
            label = f'{classNames[current_object_class_id]} ID:{int(track_id)}'
        
        
        is_violator = False
        final_box_col = (255, 0, 255)
        final_label = label
        text_params =  {'scale': 1, 'thickness': 2, 'offset': 3,'colorR':(0,0,255), 'colorB':(255,255,255), 'colorT':(0,0,0)}
        violation_type = ''
        
        #check for violations
        if current_object_class_id != -1 and current_object_class_id < len(classNames):
            if classNames[current_object_class_id] == "person" and current_light == 'green' :
                #finding center of a person at the bottom
                person_ref_x = int((tx1 + tx2) / 2)
                person_ref_y = int(ty2) 
                point_to_test = (person_ref_x, person_ref_y)
                # Check if person's point is inside the forbidden polygon
                result = cv2.pointPolygonTest(forbid_ped_zone, point_to_test, False)
                
                if result >= 0: # Inside or on the edge
                    is_violator = True
                    violation_type = 'Pedestrian'
                    final_box_col = (0, 0, 255)
                    final_label = f'Violator: {label}'
                    
            
            elif classNames[current_object_class_id] in vehicles and current_light == 'red':
                cx,cy = int((tx1+tx2)/2), int((ty1+ty2)/2) #center of a car
                point_to_test = (cx,cy) 
                # Check if car point is inside the forbidden polygon
                result = cv2.pointPolygonTest(forbid_vehicle_zone, point_to_test, False)
                
                if result >= 0: # Inside or on the edge
                    is_violator = True
                    violation_type = 'Vehicle'
                    final_box_col = (0, 0, 255)
                    final_label = f'Violator: {label}'
                    

                    
        previously_violating_flag = last_known_violation_state.get(track_id, False)
        if is_violator and not previously_violating_flag:
            timestamp_str = time.strftime("%Y%m%d-%H%M%S")
            crop_x1_save = max(0, tx1 - 20)
            crop_y1_save = max(0, ty1 - 20)
            crop_x2_save = min(img_for_detection.shape[1], tx2 + 20)
            crop_y2_save = min(img_for_detection.shape[0], ty2 + 20)
            
            if crop_x1_save < crop_x2_save and crop_y1_save < crop_y2_save:
                violator_image_crop = img_for_detection[crop_y1_save:crop_y2_save, crop_x1_save:crop_x2_save].copy() 
                if violator_image_crop.size == 0:
                    print(f"Warning: Attempted to save an empty crop for track_id:{int(track_id)}. Coordinates: x1={crop_x1_save}, y1={crop_y1_save}, x2={crop_x2_save}, y2={crop_y2_save}")
                else:
                    filename_to_save = os.path.join(img_dir, f"violator_id{int(track_id)}_{violation_type}_{timestamp_str}.jpg")
                    cv2.imwrite(filename_to_save, violator_image_crop)
                    print(f"Saved: {filename_to_save}")
            else:
                print(f"Warning: Invalid crop dimensions for track_id:{int(track_id)}. x1={crop_x1_save}, y1={crop_y1_save}, x2={crop_x2_save}, y2={crop_y2_save}")
            
            
        
        last_known_violation_state[track_id] = is_violator

        cvzone.cornerRect(img_display, (tx1, ty1, w, h), l=9, rt=2, colorR=final_box_col,colorC=final_box_col)
        cvzone.putTextRect(img_display, final_label, (max(0, tx1), max(35, ty1-10)), **text_params) 
                    
    cv2.putText(img_display, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Image', img_display)
    if cv2.waitKey(1) & 0xFF == ord('q'): # waitKey(1) for video, adjust if needed
        break

cap.release()
cv2.destroyAllWindows()





