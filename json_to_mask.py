import os
import labelme
import numpy as np
import cv2
import PIL.Image
import argparse


def json_to_mask(json_path,output_dir,image_dir):
    
    label_file=labelme.LabelFile(filename=json_path)
    # print(label_file.imagePath)
    img_path=os.path.join(image_dir,label_file.imagePath.split("\\")[-1])
    print(img_path)
    img=cv2.imread(img_path)
    (height,width)=img.shape[:2]
    img_shape=(height,width)
    mask=np.zeros(img_shape,dtype=np.uint8)
    label_names = []

    for i ,shape in enumerate(label_file.shapes):
        label=shape['label']
        points=np.array(shape['points'],dtype=np.int64)
        
        if label not in label_names:
            label_names.append(label)
        mask_img=cv2.fillPoly(mask,[points],color=(255,0,0))
    mask_img=PIL.Image.fromarray(mask.astype(np.uint8))
    mask_output_path=os.path.join(output_dir,os.path.basename(json_path).replace('.json','_mask.png'))
    mask_img.save(mask_output_path)

    with open(os.path.join(output_dir,"label_names.txt"),'w') as f:
        for lbl_name in label_names:
            f.write(lbl_name+'\n')
    print(f"Saved mask to {mask_output_path}")

ap=argparse.ArgumentParser()
ap.add_argument("-j","--json_directory",required=True,type=str,help="Path to JSON Directory")
ap.add_argument("-o","--output_directory",type=str,help="Path to output directory")
ap.add_argument('-i',"--image_directory",type=str,help="Full path to image directory")
args=vars(ap.parse_args())
json_dir=args['json_directory']
output_dir=args['output_directory']
image_dir=args["image_directory"]
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
for json_file in os.listdir(json_dir):
    if json_file.endswith('json'):
        json_path=os.path.join(json_dir,json_file)
        json_to_mask(json_path,output_dir,image_dir)
        

