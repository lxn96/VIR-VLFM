import os
import json
import numpy as np
import random
from PIL import Image
# import h5py
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate


class ClevrDataset(Dataset):

    shapes = set(['ball', 'block', 'cube', 'cylinder', 'sphere'])
    sphere = set(['ball', 'sphere'])
    cube = set(['block', 'cube'])
    cylinder = set(['cylinder'])
    colors = set(['red', 'cyan', 'brown', 'blue', 'purple', 'green', 'gray', 'yellow'])
    materials = set(['metallic', 'matte', 'rubber', 'shiny', 'metal'])
    rubber = set(['matte', 'rubber'])
    metal = set(['metal', 'metallic', 'shiny'])

    type_to_label = {
        'color': 0,
        'material': 1,
        'add': 2,
        'drop': 3,
        'move': 4,
        'no_change': 5
    }

    def __init__(self, vis_processor=None, text_processor=None, root=None, split=None):
        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.d_img_dir = os.path.join(root, 'images')
        self.s_img_dir = os.path.join(root, 'sc_images')
        self.n_img_dir = os.path.join(root, 'nsc_images')

        self.d_imgs = sorted(os.listdir(self.d_img_dir))
        self.s_imgs = sorted(os.listdir(self.s_img_dir))
        self.n_imgs = sorted(os.listdir(self.n_img_dir))

        self.splits = json.load(open(os.path.join(root, 'splits.json'), 'r'))
        self.split = split

        if split == 'train':
            self.split_idxs = self.splits['train'] + self.splits['val']
            self.num_samples = len(self.split_idxs)
        elif split == 'val':
            self.split_idxs = self.splits['val']
            self.num_samples = len(self.split_idxs)
        elif split == 'test':
            self.split_idxs = self.splits['test']
            self.num_samples = len(self.split_idxs)
        else:
            raise Exception('Unknown data split %s' % split)

        print("Dataset size for %s: %d" % (split, self.num_samples))
        self.change_caption = json.load(open(os.path.join(root, 'change_captions.json'), 'r'))
        self.no_change_caption = json.load(open(os.path.join(root, 'no_change_captions.json'), 'r'))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        random.seed()
        img_idx = self.split_idxs[index]

        # Fetch image data
        # one easy way to augment data is to use nonsemantically changed
        # scene as the default :)
        if self.split == 'train':
            if random.random() < 0.5:
                d_img_path = os.path.join(self.d_img_dir, self.d_imgs[img_idx])
                n_img_path = os.path.join(self.n_img_dir, self.n_imgs[img_idx])
            else:
                d_img_path = os.path.join(self.n_img_dir, self.n_imgs[img_idx])
                n_img_path = os.path.join(self.d_img_dir, self.d_imgs[img_idx])
        else:
            d_img_path = os.path.join(self.d_img_dir, self.d_imgs[img_idx])
            n_img_path = os.path.join(self.n_img_dir, self.n_imgs[img_idx])

        q_img_path = os.path.join(self.s_img_dir, self.s_imgs[img_idx])

        d_image = Image.open(d_img_path).convert("RGB")
        d_image = self.vis_processor(d_image)
        n_image = Image.open(n_img_path).convert("RGB")
        n_image = self.vis_processor(n_image)
        q_image = Image.open(q_img_path).convert("RGB")
        q_image = self.vis_processor(q_image)

        change_cap = self.change_caption[self.d_imgs[img_idx]]
        change_select = random.randint(0, len(change_cap) - 1)
        change_cap = change_cap[change_select]

        no_change_cap = self.no_change_caption[self.d_imgs[img_idx]]
        no_change_select = random.randint(0, len(no_change_cap) - 1)
        no_change_cap = no_change_cap[no_change_select]

        return {
            'd_image': d_image,
            'n_image': n_image,
            'q_image': q_image,
            'change_cap': change_cap,
            'no_change_cap': no_change_cap,
            'image_id': self.d_imgs[img_idx]
        }

    def get_vocab_size(self):
        return self.vocab_size

    def get_idx_to_word(self):
        return self.idx_to_word

    def get_word_to_idx(self):
        return self.word_to_idx

    def get_max_seq_length(self):
        return self.max_seq_length

def rcc_collate(batch):
    transposed = list(zip(*batch))
    d_feat_batch = transposed[0]
    n_feat_batch = transposed[1]
    q_feat_batch = transposed[2]
    seq_batch = default_collate(transposed[3])
    neg_seq_batch = default_collate(transposed[4])
    mask_batch = default_collate(transposed[5])
    neg_mask_batch = default_collate(transposed[6])
    aux_label_pos_batch = default_collate(transposed[7])
    aux_label_neg_batch = default_collate(transposed[8])
    if any(f is not None for f in d_feat_batch):
        d_feat_batch = default_collate(d_feat_batch)
    if any(f is not None for f in n_feat_batch):
        n_feat_batch = default_collate(n_feat_batch)
    if any(f is not None for f in q_feat_batch):
        q_feat_batch = default_collate(q_feat_batch)

    d_img_batch = transposed[9]
    n_img_batch = transposed[10]
    q_img_batch = transposed[11]
    return (d_feat_batch, n_feat_batch, q_feat_batch,
            seq_batch, neg_seq_batch,
            mask_batch, neg_mask_batch,
            aux_label_pos_batch, aux_label_neg_batch,
            d_img_batch, n_img_batch, q_img_batch)

class RCCDataLoader(DataLoader):
    
    def __init__(self, dataset, **kwargs):
        kwargs['collate_fn'] = rcc_collate
        super().__init__(dataset, **kwargs)
