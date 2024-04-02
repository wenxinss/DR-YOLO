# The code of DR-YOLO;


## Test Datasets used in our paper
### RTTs: please download it here:

https://sites.google.com/view/reside-dehaze-datasets

**note:** Before using the RTTs dataset, please translate it to **yolo style.**

      cd datasets
      python voc_RTTS.py

or you can download RTTs we have translated here:

https://drive.google.com/drive/folders/1O0d9Efyz1gBA3n3RXOSmR2EDRFK_1Gse?usp=drive_link

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

or 

https://drive.google.com/file/d/1nt2dOKs8NqGeq0VUCv4Lt6z5iKmboXrH/view?usp=drive_link

    python test.py --data "data/voc_fog.yaml"  --weights "best.pt"

# Train

The training data and code will also be publicly available soon. Thank you.
