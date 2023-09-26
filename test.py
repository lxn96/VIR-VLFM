import argparse
import os
import random
import json
# from tqdm import tqdm

import numpy as np
import torch
import torch.backends.cudnn as cudnn
# import gradio as gr

from vir_vlfm.common.config import Config
from vir_vlfm.common.dist_utils import get_rank
from vir_vlfm.common.registry import registry
from vir_vlfm.conversation.conversation import Chat_Clevr, CONV_VISION

# imports modules for registration
from vir_vlfm.datasets.builders import *
from vir_vlfm.models import *
from vir_vlfm.processors import *
from vir_vlfm.runners import *
from vir_vlfm.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--img1", required=True, help="path to input image 1.")
    parser.add_argument("--img2", required=True, help="path to input image 2.")
    parser.add_argument("--cfg-path", default='./eval_configs/vir_vlfm_eval.yaml',
                        help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

chat = Chat_Clevr(model, vis_processor, word_list=None, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')

# ========================================
#             Start Testing
# ========================================

user_message = "Describe the differences between this two image."
img1_path = args.img1
img2_path = args.img2

chat_state = CONV_VISION.copy()
img_list = []
llm_message = chat.upload_img(img1_path, img2_path, chat_state, img_list)
chat.ask(user_message, chat_state)

num_beams = 5
temperature = 1
do_sample = False
n_llm_message, q_llm_message = chat.answer(conv=chat_state,
                          img_list=img_list,
                          num_beams=num_beams,
                          do_sample=do_sample,
                          temperature=temperature,
                          repetition_penalty=1.5,
                          max_new_tokens=30,
                          max_length=256)

print('output_caption: {}'.format(q_llm_message))


