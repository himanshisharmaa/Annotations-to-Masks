import os
import json
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from pycocotools import mask as maskUtils
import argparse
from pathlib import Path


def create_mask(input_dir,image_dir,output_dir):
    # check if the path is file or a directory
    if os.path.isfile(annotation_path):
        ext=os.path.splitext(annotation_path)[1].lower()
        if ext==".json":
            coco_to_mask(annotation_path,image_dir,output_dir)
        else:
            raise ValueError(f"Unsupported file format for a single file: {ext}")
    
    elif os.path.isdir(annotation_path):
        for filename in os.listdir(annotation_path):
            ext=os.path.splitext(filename)[1].lower()
            file_path=os.path.join(annotation_path,filename)

            # if ext==".xml":
            #     voc_to_mask(file_path,image_dir,output_dir)
            # elif ext==".txt":
            #     yolo_to_mask(file_path,image_dir,output_dir)
            # else:
            #     print(f"Skipping unsupported file format: {filename}")
    
    else:
        raise ValueError(f"{annotation_path} is neither a valid file nor a directory")
    

def coco_to_mask(coco_ann_file,image_dir,out_dir):
    with open(coco_ann_file) as f:
        coco_data=json.load(f)
    for annotation in coco_data['annotations']:
        image_id=annotation['image_id']
        segmentation=annotation['segmentation']
        image_path=coco_data['images'][image_id-1]['file_name']
        height=coco_data['images'][image_id-1]['height']
        width=coco_data['images'][image_id-1]['width']
        full_image_path=os.path.join(image_dir,image_path)
        print(f"Height: {height}-- Width: {width}-- Image Path:{full_image_path}")
        mask=np.zeros((height,width),dtype=np.uint8)
        dir=Path(os.path.normpath(image_path)).parts
        if dir[0] in os.listdir(image_dir):
            output_directory=os.path.join(out_dir,dir[0])
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
            output_path=os.path.join(output_directory,os.path.splitext(dir[1])[0]+'.png')
        
        else:
            output_path=os.path.join(output_dir,os.path.splitext(image_path)[0]+'.png')
        print(output_path)
        
        if isinstance(segmentation,list):
            print("List")
            
            for seg in segmentation:
                poly=np.array(seg).reshape((-1,2))
                cv2.fillPoly(mask,[np.int32(poly)],color=255)
                cv2.imwrite(output_path,mask)
            cv2.waitKey(0)
        # elif isinstance(segmentation,dict) and 'counts' in segmentation:
        #     print("Dict")
        #     rle=maskUtils.frPyObjects(segmentation,height,width)
        #     mask=maskUtils.decode(rle)
       
    print(f"COCO mask saved for: {coco_ann_file}")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("-p","--input_path",type=str,required=True,help="Path to input directory or file")
    ap.add_argument("-i","--image_path",type=str,required=True,help="Path to image directory")
    ap.add_argument("-o","--output_directory",type=str,default="Outputs/",help="Path to the Output Directory")
    args=vars(ap.parse_args())

    annotation_path=args['input_path']
    output_dir=args['output_directory']
    image_dir=args['image_path']
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    create_mask(annotation_path,image_dir,output_dir)
