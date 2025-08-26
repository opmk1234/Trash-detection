Real-Time Scrap Classifier & Robotic Pick Simulation

## Overview
This project is a mini simulation of an industrial scrap sorting vision pipeline.  
It demonstrates real-time detection of recyclable materials and simulates robotic pick-point generation.

## Objectives
1. Detect recyclable waste items using a pre-trained YOLOv8 model.
2. Focus on three categories: plastic, glass, metal.
3. Build a simple conveyor simulation using a webcam or video stream.
4. Generate pick points for detected objects (center of bounding boxes).
5. Provide a basic dashboard with object counts and FPS.

## Dataset
- Dataset: TACO (Trash Annotations in Context)  
- Subset: ~150 images from categories plastic, glass, and metal  
- Format: Converted COCO annotations into YOLO format (train/val split)  

## Model
- Model: YOLOv8n (lightweight variant for faster inference)  
- Training: Fine-tuned on the subset dataset  
- Epochs: 10  
- Input size: 640x640  

## Pipeline
1. Load dataset and convert to YOLO format.  
2. Train YOLOv8n on selected categories.  
3. Run inference on webcam or video.  
4. Overlay bounding boxes and confidence scores.  
5. Generate pick-points at the center of each bounding box.  
6. Display dashboard with counts and FPS.  
