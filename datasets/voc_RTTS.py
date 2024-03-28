'''转换RTTs的标签'''
import os
import argparse
import xml.etree.ElementTree as ET
def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)
def convert_voc_annotation(data_path, data_type, anno_path, use_difficult_bbox=True):

    # classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    #            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
    #            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
    #            'train', 'tvmonitor']
    classes = ['person', 'car', 'bus', 'bicycle',  'motorbike']

    img_inds_file = os.path.join(data_path, 'ImageSets', 'Main', data_type + '.txt')
    with open(img_inds_file, 'r') as f:
        txt = f.readlines()
        image_inds = [line.strip() for line in txt]


    for image_ind in image_inds:
        ann_txt = os.path.join("/home8T/swx/yolov7/datasets/VOC_four_noVal/labels/val", image_ind +'.txt')###### notice here ###
        with open(ann_txt, 'a') as f:
            image_path = os.path.join(data_path, 'JPEGImages', image_ind + '.png')
            label_path = os.path.join(data_path, 'Annotations', image_ind + '.xml')
            root = ET.parse(label_path).getroot()
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)
            objects = root.findall('object')
            for obj in objects:
                difficult = obj.find('difficult').text.strip()
                if (not use_difficult_bbox) and(int(difficult) == 1):
                    continue
                bbox = obj.find('bndbox')
                class_ind = classes.index(obj.find('name').text.lower().strip())
                xmin = bbox.find('xmin').text.strip()
                xmax = bbox.find('xmax').text.strip()
                ymin = bbox.find('ymin').text.strip()
                ymax = bbox.find('ymax').text.strip()
                # annotation += ' ' + ','.join([str(class_ind) , xmin , ymin , xmax, ymax])
                b = (float(xmin), float(xmax), float(ymin), float(ymax))
                bb = convert((w, h), b)
                annotation = str(class_ind) + " " + " ".join([str(a) for a in bb])
                print(annotation)
                f.write(annotation + "\n")
                f.flush()

    return len(image_inds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/home8T/swx/yolov7/datasets/RTTS")
    parser.add_argument("--test_annotation",  default="/home8T/swx/yolov7/datasets/RTTS_fog/RTTTest/test.txt")
    flags = parser.parse_args()

    # if os.path.exists(flags.train_annotation):os.remove(flags.train_annotation)
    if os.path.exists(flags.test_annotation):os.remove(flags.test_annotation)
    num = convert_voc_annotation(flags.data_path,  'test', flags.test_annotation, True)
    print('=> The number of image for test is:%d' %num)