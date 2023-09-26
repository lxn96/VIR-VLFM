import os
from torch.utils.data import Dataset
import numpy as np
import json
import random
from PIL import Image


class SPOTDataset(Dataset):
    def __init__(self, vis_processor=None, text_processor=None, root=None, split=None):
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.root = root
        self.split = split
        assert self.split in ["train", "val", "test"]
        if split == 'train':
            change_caption_file = os.path.join(self.root, "annotations","reformat_%s.json" % self.split)
            change_caption_file_val = os.path.join(self.root, "annotations","reformat_val.json")
            with open(change_caption_file, 'r') as fp:
                change_captions = json.load(fp)
            with open(change_caption_file_val, 'r') as fp:
                change_captions += json.load(fp)
        else:
            change_caption_file = os.path.join(self.root, "annotations","reformat_%s.json" % self.split)
            with open(change_caption_file, 'r') as fp:
                change_captions = json.load(fp)

        self.sample_len = 0
        self.sentences_dict = {}
        self.cut_off_points = []

        for cap in change_captions:
            image_id = cap["img_id"]
            self.sentences_dict[len(self.sentences_dict)] = (image_id, cap["sentences"])
            # for cap_txt in cap["sentences"]:
            #     self.sentences_dict[len(self.sentences_dict)] = (image_id, cap_txt)
            self.cut_off_points.append(len(self.sentences_dict))

        ## below variables are used to multi-sentences retrieval
        # self.cut_off_points: used to tag the label when calculate the metric
        # self.sentence_num: used to cut the sentence representation
        # self.image_num: used to cut the image pair representation
        self.multi_sentence_per_pair = True    # !!! important tag for eval
        if self.split == "val" or self.split == "test":
            self.sentence_num = len(self.sentences_dict)
            self.image_num = len(change_captions)
            assert len(self.cut_off_points) == self.image_num
            print("For {}, sentence number: {}".format(self.split, self.sentence_num))
            print("For {}, image number: {}".format(self.split, self.image_num))

        print("Image number: {}".format(len(change_captions)))
        print("Total Paire: {}".format(len(self.sentences_dict)))
        self.sample_len = len(self.sentences_dict)

    def __len__(self):
        return self.sample_len

    def __getitem__(self, idx):
        image_id, caption = self.sentences_dict[idx]
        caption = random.choice(caption)

        d_img_path = os.path.join(self.root, 'resized_images', "%s.png" % image_id)
        q_img_path = os.path.join(self.root, 'resized_images', "%s_2.png" % image_id)
        image_idx_name = "%s.png" % image_id

        d_image = Image.open(d_img_path).convert("RGB")
        d_image = self.vis_processor(d_image)
        q_image = Image.open(q_img_path).convert("RGB")
        q_image = self.vis_processor(q_image)

        return {
            'd_image': d_image,
            # 'n_image': None,
            'q_image': q_image,
            'change_cap': caption,
            # 'no_change_cap': None,
            'image_id': image_idx_name
        }
