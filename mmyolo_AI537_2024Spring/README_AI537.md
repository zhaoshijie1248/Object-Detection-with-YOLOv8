## Preparing for install
To get setup, we first need to transfer the project onto the HPC and request GPU resources for the installation and compilation of MMYOLO and required packages.

First transfer the downloaded project from your local machine to the HPC server:
```
scp {local/path/to}/mmyolo_AI537_2024Spring.zip {YOUR_ONID}@submit-a.hpc.engr.oregonstate.edu:/nfs/hpc/share/{YOUR_ONID}/mmyolo_AI537_2024Spring.zip
```

Then ssh onto the HPC:
```
ssh {YOUR_ONID}@submit-a.hpc.engr.oregonstate.edu
```

Load the slurm module:
```
module load slurm
```

Request a GPU on the HPC via slurm:
```
srun -A ai537 -p class --gres=gpu:1 --pty bash -i
```

Set needed environment variables:
```
module load gcc/9.5
module load cuda/11.8
```

Double check you have a GPU and that the CUDA/CUDNN libraries are detected:
```
nvidia-smi
nvcc --version
```


## Install mmyolo and dependency libraries
We will now install MMYOLO and the dependency libraries using the following commands.

If you are interested in learning more about configuring the MMYOLO environment, see [Installation and verification](https://mmyolo.readthedocs.io/en/latest/get_started/installation.html).

Create a new conda environment and install MMYOLO and its dependencies:
```shell
# create new conda environment
conda create --name myenv_hw3 python=3.8 -y
conda activate myenv_hw3

# install Pytorch
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
python -c 'import torch;print(torch.__version__)'

# install dependency libraries
cd /nfs/hpc/share/{YOUR_ONID}
unzip mmyolo_AI537_2024Spring.zip
cd mmyolo_AI537_2024Spring
pip install -U openmim
mim install -r requirements/mminstall.txt
mim install -r requirements/albu.txt

# compile mmyolo
mim install -v -e .

# install jupyter lab
pip install jupyterlab jupyter
```


## Setting up the dataset
The cat_dog_monkey dataset is a three category dataset. The original dataset is from 
[Monkey, Cat and Dog detection](https://www.kaggle.com/datasets/tarunbisht11/yolo-animal-detection-small).

We have done some processing of this dataset to make it suitable for this assignment. You will download the processed data from the HW 3 assignment page on Canvas and place the directory _cat\_dog\_monkey\_dataset/_ under the directory _data/_ of the _mmyolo\_AI537\_2024Spring_ project. 

The data should have the following structure in the project directory:
```
- mmyolo_AI537_2024Spring/
    - data/
        - cat_dog_monkey_dataset/
            - train/
                - cats_001.jpg
                - cats_002.jpg
                ...
            - val/
                - cats_000.jpg
                - cats_007.jpg
                ...
            - train_cat_annotation_100.json
            - train_cat_annotation_50.json
            - train_cat_annotation_20.json
            - train_cat_annotation_5.json
            - train_cat_annotation_0.json
            - val.json
```

## Updating YOLOv8 config for training/testing
For this assignment, the only place where changes need to be made is in _configs/yolov8/ai537_yolov8_config.py_ (and optionally the provided juypter notebook _ai537_object_detection.ipynb_).

We have constructed several datasets that contain varying amounts of labeled/annotated cats in the training set. You will be training models with these different amounts of annotated data and comparing the test performance of these models. For this, you will have to use our different annotation files (located in _data/cat_dog_monkey_dataset/_) in your training:
- train_cat_annotation_100.json (all cat bounding box annotations are provided in training set)
- train_cat_annotation_50.json (50% of cat bounding box annotations from the original dataset are provided in training set)
- train_cat_annotation_20.json (20% of cat bounding box annotations from the original dataset are provided in training set)
- train_cat_annotation_5.json (5% of cat bounding box annotations from the original dataset are provided in training set)
- train_cat_annotation_0.json (no cat bounding box annotations are provided in training set)

To specify which dataset is being used for a training run, you can modify the __train_dataloader__ settings in our config file _configs/yolov8/ai537_yolov8_config.py_:
```python
# training data loader settings
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train_cat_annotation_100.json',     # <----- Modify this line with the different "train_cat_annotation_*.json" files
        data_prefix=dict(img='train/')))
```

Additionally, for each training dataset we have provided, you will be trying to find the combination of losses and loss weights that produces the best performance on the provided validation dataset.

To specify the losses being used and the weight of each loss, you can modify the __loss\_*__ settings in the __model__ settings of our config file _configs/yolov8/ai537_yolov8_config.py_:
```python
# weighting of loss functions (you will play around with setting different values for the losses here)
loss_cls_weight = 1.0
loss_bbox_weight = 7.5
loss_dfl_weight = 0.375

# model parameter settings
model = dict(
    backbone=dict(frozen_stages=4),
    bbox_head=dict(
        head_module=dict(
            num_classes=num_classes),
        # classification loss
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='none',
            loss_weight=loss_cls_weight),     # play around with different values for loss_cls_weight
        # loss_cls=dict(
        #     type='mmdet.FocalLoss',     # uncomment this loss_cls and comment out the one above to switch from Cross Entropy to Focal loss
        #     use_sigmoid=True,
        #     reduction='none',
        #     gamma=2.0,
        #     alpha=0.25,
        #     loss_weight=loss_cls_weight),     # play around with different values for loss_cls_weight
        # bounding box regression loss
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='ciou',     # play around with different iou_mode = ['ciou' | 'giou']
            bbox_format='xyxy',
            reduction='sum',
            loss_weight=loss_bbox_weight,     # play around with different values for loss_bbox_weight
            return_iou=False),
        # distribution focal loss 
        loss_dfl=dict(
            type='mmdet.DistributionFocalLoss',
            reduction='mean',
            loss_weight=loss_dfl_weight)),     # play around with different values of loss_dfl_weight
    train_cfg=dict(assigner=dict(num_classes=num_classes)))
```

You will be expected to find the best combination of losses for the three different losses used in YOLOv8:
- loss_cls (classification loss)
- loss_bbox (bounding box regression loss)
- loss_dfl (distribution focal loss)

### loss_cls
__Relation to assignment description:__ _loss\_cls_ is the classification loss and corresponds to either $L_{CE}$ or $L_{F}$ from the assignment description on Canvas. Which loss it represents depends on which _loss\_cls_ is uncommented. If the _loss\_cls_ with _type='mmdet.CrossEntropyLoss'_ is uncommented, then we are using the [Cross Entropy Loss](https://mmdetection.readthedocs.io/en/v2.9.0/api.html#mmdet.models.losses.CrossEntropyLoss) and _loss\_cls_ corresponds to $L_{CE}$ and _loss\_cls\_weight_ to $\lambda_{CE}$. Alternatively, if the _loss\_cls_ with _type='mmdet.FocalLoss'_ is uncommented, then we are using the [Focal Loss](https://mmdetection.readthedocs.io/en/v2.9.0/api.html#mmdet.models.losses.FocalLoss) and _loss\_cls_ corresponds to $L_{F}$ and _loss\_cls\_weight_ to $\lambda_{F}$. Only one of these should be uncommented at any time, so you are only using one of the losses $L_{CE}$ or $L_{F}$ (even if both are uncommented only the later of the two will be used).

### loss_bbox
__Relation to assignment description:__ _loss\_bbox_ is the bounding box regression loss (IoU loss function) and corresponds to either $L_{CIoU}$ or $L_{GIoU}$ from the assignment description on Canvas. Which loss it represents depends on what _iou\_mode_ in _loss\_bbox_ is set to. If _iou\_mode='ciou'_, then we are using the Complete Intersection over Union (CIoU) and _loss\_bbox_ corresponds to $L_{CIoU}$ and _loss\_bbox\_weight_ to $\lambda_{CIoU}$. Alternatively, If _iou\_mode='giou'_, then we are using the Generalized Intersection over Union (GIoU) and _loss\_bbox_ corresponds to $L_{GIoU}$ and _loss\_bbox\_weight_ to $\lambda_{GIoU}$.

### loss_dfl
__Relation to assignment description:__  _loss\_dfl_ is the distribution focal loss and corresponds to $L_{DF}$ from the assignment description on Canvas. Similarly, the parameter _loss\_dfl\_weight_ corresponds to $\lambda_{DF}$.