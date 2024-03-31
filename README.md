# The code of DR-YOLO;


## Test Datasets used in our paper
### RTTs: please download it here:

https://sites.google.com/view/reside-dehaze-datasets

**note:** Before using the RTTs dataset, please translate it to **yolo style.**

      cd datasets
      python voc_RTTS.py

### VF-test：please download it here:

https://github.com/wenyyu/Image-Adaptive-YOLO

### VN-test： please download the VOC2007_test here:

http://host.robots.ox.ac.uk/pascal/VOC/voc2007/

**note:** Before using the VOC2007_test dataset, please filter the five classes in our paper and translate it to **yolo style.**

      cd datasets
      python voc_annotation.py
  
# Test
## The best weightfile in our paper is here:

https://pan.baidu.com/s/1LasrDnfKE-wY5IuV_FG_QQ?pwd=ssss 

    python test.py --data "data/voc_fog.yaml"  --weights "best.pt"
