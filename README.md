This is an unofficial implement of multi-label image classification model C-Tran.

[[paper]](https://arxiv.org/abs/2011.14027)
[[official repo]](https://github.com/QData/C-Tran#readme)

Datasets used in this implement are COCO-80 and VOC. COCO-80 uses `imdb` to load. The create method is provided.

The whole training and test progress are re-written with pytorch and pytorch_lightning.

## Install

```bash
# clone this repo

pip3 install -r requirement.txt

python3 train.py --logger comet --apikey {your_comet_apikey} --voc_root {your_voc_dataset_dir} --gpus 0 --experiment voc --run nice_run_name --mask_rate 0.5 --epoch 100 --batch_size 128

# no logger
python3 train.py --logger dummy  --coco_lmdb {your_coco-lmdb_path} --gpus 1 --experiment coco --coco_train_json {coco_train_annotation_json_path} --coco_val_json {coco_val_annotation_json_path} --run nice_run_name --mask_rate 0.5 --epoch 100 --batch_size 128

```

## Citing
```bibtex
@article{lanchantin2020general,
  title={General Multi-label Image Classification with Transformers},
  author={Lanchantin, Jack and Wang, Tianlu and Ordonez, Vicente and Qi, Yanjun},
  journal={arXiv preprint arXiv:2011.14027},
  year={2020}
}
```
