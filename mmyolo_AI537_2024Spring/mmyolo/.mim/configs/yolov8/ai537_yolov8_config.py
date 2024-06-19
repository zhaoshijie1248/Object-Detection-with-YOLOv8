# base config to inherit most settings from
_base_ = 'yolov8_s_syncbn_fast_8xb16-500e_coco.py'

# dataset info
data_root = './data/cat_dog_monkey_dataset/'
class_name = ('cat', 'dog', 'monkey')
num_classes = len(class_name)
metainfo = dict(classes=class_name, palette=[(20, 220, 60), (119, 11, 32), (165, 42, 42)])

# training hyperparameters
max_epochs = 20
train_batch_size_per_gpu = 8
close_mosaic_epochs = 5

# number of worker processes for dataloader (set <= # of CPU cores requested)
train_num_workers = 1

# weighting of loss functions (you will play around with setting different values for the losses here)
loss_cls_weight = 1.0
loss_bbox_weight = 7.5
loss_dfl_weight = 0.375

# pretrained weights for YOLOv8 backbone
load_from = 'https://download.openmmlab.com/mmyolo/v0/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1.pth'  # noqa

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

# training data loader settings
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train_cat_annotation_100.json',     # <----- Modify this line with the different "train_cat_annotation_*.json" files      
        data_prefix=dict(img='train/')))

# validation data loader settings
val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='val.json',
        data_prefix=dict(img='val/')))

# extra settings
test_dataloader = val_dataloader
_base_.optim_wrapper.optimizer.batch_size_per_gpu = train_batch_size_per_gpu
_base_.custom_hooks[1].switch_epoch = max_epochs - close_mosaic_epochs
val_evaluator = dict(ann_file=data_root + 'val.json')
test_evaluator = val_evaluator
default_hooks = dict(
    checkpoint=dict(interval=5, max_keep_ckpts=2, save_best='auto'),
    # The warmup_mim_iter parameter is critical.
    # The default value is 1000 which is not suitable for cat datasets.
    param_scheduler=dict(max_epochs=max_epochs, warmup_mim_iter=10),
    logger=dict(type='LoggerHook', interval=5))
train_cfg = dict(max_epochs=max_epochs, val_interval=10)
# visualizer = dict(vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend')]) # noqa
