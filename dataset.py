import os
import lmdb
import tqdm
import torch
import torchvision
import torch.utils.data
from PIL import Image
import io
from typing import *
import json
import pickle

COCO_LABEL_INDEX2ID = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
COCO_LABEL_ID2INDEX = {j:i for i, j in enumerate(COCO_LABEL_INDEX2ID)}

class COCOlmdb(torch.utils.data.Dataset):
    def __init__(
        self, 
        lmdb_path: str, 
        mode_json_paths: Dict[str, str], 
        mode: str, 
        transform: torch.nn.Module = torchvision.transforms.PILToTensor()
    ):
        assert mode in mode_json_paths, "assigned mode should be in mode json dict"
        self.env = lmdb.open(lmdb_path)
        self.tnx = self.env.begin()
        self.mode_json_paths = mode_json_paths
        self.transform = transform
        self.change_mode(mode)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        img_name = self.iter_key[index]
        im = Image.open(io.BytesIO(self.tnx.get(img_name.encode('utf-8'))))
        im = self.transform(im)
        img_id = int(img_name.split('_')[-1].split('.')[0])
        img_id = self.id_mapping[img_id]
        label = self.label[img_id]
        return im, label
        

    def __len__(self) -> int:
        return len(self.id_mapping)
    
    def __del__(self):
        self.env.close()

    def change_mode(self, mode: str):
        self.mode = mode
        assert self.mode in self.mode_json_paths, "assigned mode should be in mode json dict"
        tensor_path = '.'.join(self.mode_json_paths[self.mode].split('.')) + '_lmdb.pkl'
        if os.path.isfile(tensor_path):
            with open(tensor_path, 'rb') as f:
                self.iter_key, self.id_mapping, self.label = pickle.load(f)
        else:
            anno_json = json.load(open(self.mode_json_paths[self.mode], 'r'))
            self.id_mapping = {j: i for i, j in enumerate(sorted(i['id'] for i in anno_json['images']))}
            self.iter_key = [i['file_name'] for i in anno_json['images']]
            self.label = torch.zeros(len(self.id_mapping), 80)
            for anno in anno_json['annotations']:
                # if anno['image_id'] in self.id_mapping:
                self.label[self.id_mapping[anno['image_id']]][COCO_LABEL_ID2INDEX[anno['category_id']]] = 1.0
            with open(tensor_path, 'wb') as f:
                pickle.dump((self.iter_key, self.id_mapping, self.label), f)
        


def create_lmdb(
        lmdb_path: str,
        dataset_dir: str
    ):
    COMMIT_INTERNVAL = 16
    with lmdb.open(lmdb_path, map_size=257698037760) as env:
        cnt = 0
        tnx = env.begin(write=True)
        for i, j, k in os.walk(dataset_dir):
            for img_name in tqdm.tqdm(k):
                if any(img_name.lower().endswith(ending) for ending in (".jpg", ".png", ".bmp")):
                    img_path = os.path.join(i, img_name)
                    with open(img_path, 'rb') as f:
                        img_byte = f.read()
                    tnx.put(img_name.encode("utf-8"), img_byte)
                    cnt += 1
                    if cnt == COMMIT_INTERNVAL:
                        cnt = 0
                        tnx.commit()
                        tnx = env.begin(write=True)
        tnx.commit()

def _iter_lmdb(imdb_path):
    with lmdb.open(imdb_path) as env:
        txn = env.begin()
        cur = txn.cursor()
        for k, v in tqdm.tqdm(cur):
            img = Image.open(io.BytesIO(v))
            # print(k.decode('utf-8'), img.size)

class VOCDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        root: Optional[str]='./voc/detection',
        transform: Optional[torch.nn.Module]=None,
        partition: str = 'train',
    ) -> None:
        super().__init__()
        self.dataset = torchvision.datasets.VOCDetection(root, image_set=partition)
        self.transform = transform if transform else torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=torch.tensor([0.4571, 0.4383, 0.4062]),
                std=torch.tensor([0.2708, 0.2680, 0.2817]),
            ),
        ])
        self.idx2label = [
            "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", 
            "car", "cat", "chair", "cow", "diningtable", "dog", "horse", 
            "motorbike", "person", "pottedplant", "sheep", "sofa", "train", 
            "tvmonitor"
        ]
        self.label2idx = {j: i for i, j in enumerate(self.idx2label)}
    
    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> Any:
        img, anno = self.dataset[idx]
        img = self.transform(img)
        anno = [self.label2idx[n["name"]] for n in anno["annotation"]["object"]]
        label_idx = torch.zeros(len(self.idx2label))
        label_idx[torch.tensor(anno)] = 1.0
        return img, label_idx
        