import argparse
from pytorch_lightning.loggers.mlflow import MLFlowLogger
from pytorch_lightning.loggers.base import DummyLogger
from pytorch_lightning.loggers.comet import CometLogger
import pytorch_lightning as pl
import pl_model
import pl_dataset
from pytorch_lightning.callbacks import ModelCheckpoint

parser = argparse.ArgumentParser()
parser.add_argument("--logger", type=str, default="", choices=["dummy", "mlflow", "comet"], help="logger type, one of 'dummy', 'mlflow', 'comet'")
parser.add_argument("--apikey", type=str, default="", help="api key for comet")
parser.add_argument("--gpus", type=str, default="", help="gpu ids, comma separated")
parser.add_argument("--experiment", type=str, default="", help="experiment name")
parser.add_argument("--run", type=str, default="", help="run name")
parser.add_argument("--voc_root", type=str, default="", help="voc dataset root, will be used if experiment contains voc")
parser.add_argument("--coco_lmdb", type=str, default="", help="coco lmdb dataset root, will be used if experiment contains coco")
parser.add_argument("--coco_train_json", type=str, default="", help="coco train json, will be used if experiment contains coco")
parser.add_argument("--coco_val_json", type=str, default="", help="coco val json root, will be used if experiment contains coco")
parser.add_argument("--run", type=str, default="", help="run name")
parser.add_argument("--mask_rate", type=float, default=0.5, help="mask rate")
parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument("--epoch", type=int, default=100, help="number of epoch")
args = parser.parse_args()

pl.seed_everything(1024)


if 'coco' in args.experiment:
    model = pl_model.ModelWrapper(
        backbone_name='resnet101',
        num_classes=80,
        lrearning_rate= 1e-6,
        num_encoder_layers=3,
        mask_rate=args.mask_rate,
    )

    dataset_module = pl_dataset.COCO(
        lmdb_path=args.coco_lmdb, 
        mode_json_paths={
            "train": args.coco_train_json,
            "val": args.coco_val_json,
        },
        batch_size=args.batch_size
    )

elif 'voc' in args.experiment:
    model = pl_model.ModelWrapper(
        backbone_name='resnet101',
        num_classes=20,
        lrearning_rate= 1e-6,
        num_encoder_layers=3,
        mask_rate=args.mask_rate,
    )

    dataset_module = pl_dataset.VOC(
        root=args.voc_root,
        batch_size=args.batch_size
    )

else:
    raise ValueError("experiment should contain one of 'voc', 'coco'")

if args.logger == 'dummy':
    logger = DummyLogger()
elif args.logger == 'mlflow':
    logger = MLFlowLogger(
        experiment_name=args.experiment,
        run_name=args.run,
    )
elif args.logger == 'comet':
    logger = CometLogger(
        api_key=args.apikey,
        project_name=args.experiment,
        experiment_name=args.run,
    )
else:
    raise ValueError("logger should be one of 'dummy', 'mlflow', 'comet'")

trainer = pl.Trainer(
    logger=logger,
    max_epochs=100,
    precision=16,
    accelerator='gpu', 
    devices=[int(i) for i in args.gpus.split(',')],
    # val_check_interval=0.3,
    callbacks=[
        ModelCheckpoint(
            dirpath='./cache/ctran',
            save_last=True,
            save_top_k=3,
            monitor='mAP'
        )
    ]
    )

trainer.fit(model, dataset_module)
