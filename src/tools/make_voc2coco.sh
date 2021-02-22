for split in train val trainval test
do
    python voc2coco.py \
        --ann_dir ~/datasets/VOC2007/Annotations \
        --ann_ids ~/datasets/VOC2007/ImageSets/Main/${split}.txt \
        --labels ~/datasets/VOC2007/labels.txt \
        --output ~/datasets/VOC2007/coco_labels/${split}.json \
        --ext xml
done
