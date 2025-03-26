import glob
import os
import sys

import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

#Data dir relative to project root
BASE_ROOT = "/cluster/home/mathialm/poisoning/ML_Poisoning/data"

#Since we only consider unconditional text to image, all text will be "" so we dont need it
class Base(Dataset):
    def __init__(self,
                 data_root,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5
                 ):
        self.data_root = data_root
        self.image_paths = [os.path.basename(f) for f in glob.glob(pathname=os.path.join(self.data_root, "*.png"))]

        self._length = len(self.image_paths)
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l)
                           for l in self.image_paths],
        }

        self.size = size
        self.interpolation = {"bilinear": PIL.Image.Resampling.BILINEAR,
                              "bicubic": PIL.Image.Resampling.BICUBIC,
                              "lanczos": PIL.Image.Resampling.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        return example


class Train(Base):
    def __init__(self, attack, **kwargs):
        ATTACK_ROOT = os.path.join(BASE_ROOT, "datasets32", "COCO_TRAFFICext")
        attack_dirs = os.listdir(ATTACK_ROOT)
        print(attack_dirs)
        sys.exit(0)

        assert attack in attack_dirs

        super().__init__(data_root=os.path.join(BASE_ROOT, "datasets32", "COCO_TRAFFICext", attack, "train"), **kwargs)


class Val(Base):
    def __init__(self, attack, flip_p=0., **kwargs):
        assert attack in ATTACKS
        super().__init__(data_root=os.path.join(BASE_ROOT, "datasets32", "COCO_TRAFFICext", attack, "val"), flip_p=flip_p, **kwargs)