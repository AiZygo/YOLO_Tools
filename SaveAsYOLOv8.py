# SaveAsYOLOv8.py
# This script provides functions to create YOLOv8 dataset folders, save class names in a YAML file,
# convert bounding box coordinates to YOLO format, and save images and annotations in the YOLO format. 
# It also includes a function to split a dataset into training and validation sets.

import os
import yaml
import cv2
import random
import shutil

def makeyolofolders(savepath='../'):
    # Create directories if they don't exist
    os.makedirs(savepath+"train/images", exist_ok=True)
    os.makedirs(savepath+"train/labels", exist_ok=True)
    os.makedirs(savepath+"valid/images", exist_ok=True)
    os.makedirs(savepath+"valid/labels", exist_ok=True)

def makedatayaml(class_names,savepath='../'):
    # Save class names to data.yaml
    with open(savepath+'data.yaml', "w") as file2:
        file2.write('train: ../train/images'+'\n'+
                    'val: ../valid/images'+'\n'+
                    '\n'+ 
                    'nc: '+str(len(class_names))+'\n'+
                    'names: ['+str(class_names)[1:-1]+']')

def bbox_to_yolo_format(xmin, ymin, xmax, ymax, original_width, original_height, target_size=640, decimal_places=5):
    # Calculate scale factors
    x_scale = target_size / original_width
    y_scale = target_size / original_height
    
    # Resize bounding box coordinates
    xmin_resized = xmin * x_scale
    ymin_resized = ymin * y_scale
    xmax_resized = xmax * x_scale
    ymax_resized = ymax * y_scale
    
    # Calculate the center coordinates
    x_center = (xmin_resized + xmax_resized) / 2
    y_center = (ymin_resized + ymax_resized) / 2
    
    # Calculate the width and height
    bbox_width = xmax_resized - xmin_resized
    bbox_height = ymax_resized - ymin_resized
    
    # Normalize the coordinates
    x_center /= target_size
    y_center /= target_size
    bbox_width /= target_size
    bbox_height /= target_size
    
    # Round the coordinates to the specified number of decimal places
    x_center = round(x_center, decimal_places)
    y_center = round(y_center, decimal_places)
    bbox_width = round(bbox_width, decimal_places)
    bbox_height = round(bbox_height, decimal_places)
    
    # Return the YOLO format: [x_center, y_center, bbox_width, bbox_height]
    return [x_center, y_center, bbox_width, bbox_height]

def polygon_to_yolo_format(polygon_points, original_width, original_height, target_size=640, decimal_places=5):
    # Calculate scale factors
    x_scale = target_size / original_width
    y_scale = target_size / original_height
    
    # Resize and normalize polygon coordinates
    yolo_polygon = []
    for (x, y) in polygon_points:
        x_resized = x * x_scale
        y_resized = y * y_scale
        x_normalized = x_resized / target_size
        y_normalized = y_resized / target_size
        yolo_polygon.append((round(x_normalized, decimal_places), round(y_normalized, decimal_places)))
    
    # Flatten the list of tuples into a single list of coordinates
    yolo_polygon_flat = [coord for point in yolo_polygon for coord in point]
    
    return yolo_polygon_flat

def saveasyolo(cv2img, classindex, xypoints, frame_number,folder,yolo_img_size=640):
  
    
    # Convert bounding box coordinates to YOLO format
    imgW=cv2img.shape[1]
    imgH=cv2img.shape[0]
    xywh=bbox_to_yolo_format(int(xypoints[0][0]),int(xypoints[0][1]),int(xypoints[1][0]),int(xypoints[1][1]),imgW,imgH)
     
    # Create YOLO annotation string
    yolo_annotation = str(classindex)+" "+ ' '.join(map(str, xywh))+" \n"

    resizedimg=cv2.resize(cv2img,[yolo_img_size,yolo_img_size])

    # Save image
    image_filename = f"{folder}/{frame_number}.jpg"
    #image_filename = f"{folder}/images/{frame_number}.jpg"
    cv2.imwrite(image_filename, resizedimg)

    # Save YOLO annotation
    label_filename = f"{folder}/{frame_number}.txt"
    #label_filename = f"{folder}/labels/{frame_number}.txt"

    with open(label_filename, "a") as label_file:
        label_file.write(yolo_annotation + "\n")

def makeyolo(source_directory,ratio=0.8):
    # Define the destination directories
    train_image_dir = source_directory+"train/images"
    train_label_dir =source_directory+ "train/labels"
    valid_image_dir =source_directory+ "valid/images"
    valid_label_dir =source_directory+ "valid/labels"

    # Get list of all JPG files in the source directory
    jpg_files = [file for file in os.listdir(source_directory) if file.endswith(".jpg")]

    # Create a list to hold file pairs (JPG and corresponding TXT)
    file_pairs = []

    # Populate file pairs list
    for jpg_file in jpg_files:
        txt_file = jpg_file.replace(".jpg", ".txt")
        if txt_file in os.listdir(source_directory):
            file_pairs.append((jpg_file, txt_file))

    # Shuffle the file pairs randomly
    random.shuffle(file_pairs)

    # Calculate the number of pairs for training and validation
    num_train_pairs = int(ratio * len(file_pairs))
    num_valid_pairs = len(file_pairs) - num_train_pairs

    # Split file pairs into training and validation sets
    train_pairs = file_pairs[:num_train_pairs]
    valid_pairs = file_pairs[num_train_pairs:]

    # Move files to their respective directories
    for jpg_file, txt_file in train_pairs:
        shutil.move(os.path.join(source_directory, jpg_file), os.path.join(train_image_dir, jpg_file))
        shutil.move(os.path.join(source_directory, txt_file), os.path.join(train_label_dir, txt_file))

    for jpg_file, txt_file in valid_pairs:
        shutil.move(os.path.join(source_directory, jpg_file), os.path.join(valid_image_dir, jpg_file))
        shutil.move(os.path.join(source_directory, txt_file), os.path.join(valid_label_dir, txt_file))

