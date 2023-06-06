import os
from PIL import Image
from torch.utils.data import Dataset
import json
import nori2 as nori
import cv2
import io
import numpy as np

class ImageList(object):

    def __init__(self, root, list_file, select=False):
        if isinstance(list_file, str):
            with open(list_file, 'r') as f:
                lines = f.readlines()
            self.has_labels = len(lines[0].split()) == 2
            if self.has_labels:
                self.fns, self.labels = zip(*[l.strip().split() for l in lines])
                self.labels = [int(l) for l in self.labels]
            else:
                self.fns = [l.strip() for l in lines]
        elif isinstance(list_file, list):
            self.has_labels = len(list_file[0]) == 2
            if self.has_labels:
                self.fns, self.labels = zip(*list_file)
            else:
                self.fns = list_file
        
        if select:
            assert self.has_labels
            n_fns = []
            n_labels = []
            cls_cnt_dict = {}
            for fns, label in zip(self.fns, self.labels):
                if label not in cls_cnt_dict:
                    cls_cnt_dict[label] = 0
                cls_cnt_dict[label] += 1
                if cls_cnt_dict[label] > 50: continue
                n_fns.append(fns)
                n_labels.append(label)
            self.fns = n_fns
            self.labels = n_labels
        
        self.fns = [os.path.join(root, fn) for fn in self.fns]

    def get_length(self):
        return len(self.fns)

    def get_sample(self, idx):
        img = Image.open(self.fns[idx])
        img = img.convert('RGB')
        if self.has_labels:
            target = self.labels[idx]
            return img, target
        else:
            return img


class BaseLTnoriDataset(Dataset):
    def __init__(
        self, ann_paths=[], select=False
    ):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """

        self.annotation = []
        for ann_path in ann_paths:
            self.annotation.extend(json.load(open(ann_path, "r"))['annotations'])

        path2id = json.load(open(os.path.dirname(ann_path) + '/' + 'path2id.json', "r"))
        
        for ann in self.annotation:
           ann['nori_id'] = path2id[ann['fpath']]

        self.nori_fetcher = None
        self.labels = [int(ann['category_id']) for ann in self.annotation]
        if select:
            n_anns = []
            n_labels = []
            cls_cnt_dict = {}
            for label in self.labels:
                if label not in cls_cnt_dict:
                    cls_cnt_dict[label] = 0
                cls_cnt_dict[label] += 1
                if cls_cnt_dict[label] > 50: continue
                n_anns.append(ann)
                n_labels.append(int(ann['category_id']))
            self.annotation = n_anns
            self.labels = n_labels

    def __len__(self):
        return len(self.annotation)

    def get_length(self):
        return len(self.annotation)

    def _check_nori_fetcher(self):
        """Lazy initialize nori fetcher. In this way, `NoriDataset` can be pickled and used
            in multiprocessing.
        """
        if self.nori_fetcher is None:
            self.nori_fetcher = nori.Fetcher()

    def __getitem__(self, index):

        self._check_nori_fetcher()
        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        nori_id = ann["nori_id"]
        img_bytes = self.nori_fetcher.get(nori_id)
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        except:
            img = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)

        img_label = ann['category_id']  # 0-index

        return img, img_label
