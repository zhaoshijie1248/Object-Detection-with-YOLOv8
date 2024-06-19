## This script is used to conver the xml files into json file for mmyolo
## written by Liqiang He, 20240328

import os.path as osp
import os
import mmcv
import mmengine
import xml.etree.ElementTree as ET
import json

category_set = {'cat':0, 'dog':1, 'monkey':2}
def xml_to_mmyolo(xml_file, img_idx, obj_count):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # annotation = {
    #     "filename": root.find('filename').text,
    #     "width": int(root.find('size/width').text),
    #     "height": int(root.find('size/height').text),
    #     "depth": int(root.find('size/depth').text),
    #     "objects": []
    # }
    annotations = []
    for obj in root.findall('object'):
        x_min = int(obj.find('bndbox/xmin').text)
        y_min = int(obj.find('bndbox/ymin').text)
        x_max = int(obj.find('bndbox/xmax').text)
        y_max = int(obj.find('bndbox/ymax').text)
        category_id = category_set[obj.find('name').text]
        data_anno = dict(
            image_id=img_idx,
            id=obj_count,
            category_id=category_id,
            bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
            area=(x_max - x_min) * (y_max - y_min),
            segmentation=[],
            iscrowd=0)  
        obj_count += 1
        annotations.append(data_anno)

    return obj_count, annotations

def convert_catDogMonkey_to_coco(xml_dir, output_file):
    print("Begin to convert {} to {}".format(xml_dir, output_file))
    annotations = []
    images = []
    obj_count = 0
    xml_files = [ele for ele in os.listdir(xml_dir) if ele.endswith('.xml')]
    
    for img_idx, xml_file in enumerate(mmengine.track_iter_progress(xml_files)):
        filename = xml_file[:-4] + '.jpg'
        xml_path = osp.join(xml_dir, xml_file)
        img_path = osp.join(xml_dir, filename)
        if not osp.exists(img_path):
            print("image-xml pair does not exist for {} and {}!".format(img_path, xml_path))
            exit()
            continue
        height, width = mmcv.imread(img_path).shape[:2]
        images.append(
            dict(id=img_idx, file_name=filename, height=height, width=width))
        # go through all the objets on this image
        obj_count, annotation = xml_to_mmyolo(xml_path, img_idx=img_idx, obj_count=obj_count)
        annotations.extend(annotation)

    print("Begin to write to {}".format(output_file))
    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=[
            {'id': 0,
            'name': 'cat'},
            {'id': 1,
            'name': 'dog'},
            {'id': 2,
            'name': 'monkey'},
            ])
    
    mmengine.dump(coco_format_json, output_file)
    print("Done!")
    

if __name__ == '__main__':

    convert_catDogMonkey_to_coco('data/cat_dog_monkey_dataset/train/',
                            'data/cat_dog_monkey_dataset/train.json')
    convert_catDogMonkey_to_coco('data/cat_dog_monkey_dataset/val/',
                            'data/cat_dog_monkey_dataset/val.json')
    convert_catDogMonkey_to_coco('data/cat_dog_monkey_dataset/test/',
                            'data/cat_dog_monkey_dataset/test.json')
