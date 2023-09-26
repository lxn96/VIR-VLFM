import argparse
import time
from PIL import Image
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList

import dataclasses
from enum import auto, Enum
from typing import List, Tuple, Any
from vir_vlfm.common.registry import registry


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    # system_img: List[Image.Image] = []
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None

    skip_next: bool = False
    conv_id: Any = None

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            # system_img=self.system_img,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            conv_id=self.conv_id)

    def dict(self):
        return {
            "system": self.system,
            # "system_img": self.system_img,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "conv_id": self.conv_id,
        }


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False


CONV_VISION = Conversation(
    system="Give the following image: <Img>ImageContent</Img>. "
           "You will be able to see the image once I provide it to you. Please answer my questions.",
    roles=("Human", "Assistant"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)


class Chat_Clevr:
    def __init__(self, model, vis_processor, word_list=None, device='cuda:0'):
        self.device = device
        self.model = model
        self.vis_processor = vis_processor
        stop_words_ids = [torch.tensor([835]).to(self.device),
                          torch.tensor([2277, 29937]).to(self.device)]  # '###' can be encoded in two different ways.
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

        self.word_list = word_list
        if word_list is not None:
            self.word_list_ids = self.model.llama_tokenizer(word_list, add_special_tokens=False).input_ids
            gt_vocab = []
            for items in self.word_list_ids:
                for item in items:
                    gt_vocab.append(item)
            gt_vocab.append(835)
            gt_vocab.append(2277)
            gt_vocab.append(29937)
            word_new = []
            for i in range(self.model.llama_model.model.vocab_size):
                if i not in gt_vocab:
                    word_new.append([i])
            self.bad_words_ids = word_new

    def ask(self, text, conv):
        if len(conv.messages) > 0 and conv.messages[-1][0] == conv.roles[0] \
                and conv.messages[-1][1][-6:] == '</Img>':  # last message is image.
            conv.messages[-1][1] = ' '.join([conv.messages[-1][1], text])
        else:
            conv.append_message(conv.roles[0], text)

    def answer(self, conv, img_list, max_new_tokens=300, num_beams=1, do_sample=True, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, max_length=2000):
        conv.append_message(conv.roles[1], None)
        n_embs, q_embs = self.get_context_emb(conv, img_list)

        if n_embs is not None:
            n_current_max_len = n_embs.shape[1] + max_new_tokens
            if n_current_max_len - max_length > 0:
                print('Warning: The number of tokens in current conversation exceeds the max length. '
                    'The model will not see the contexts outside the range.')
            n_begin_idx = max(0, n_current_max_len - max_length)
            n_embs = n_embs[:, n_begin_idx:]
            if self.word_list is not None:
                n_outputs = self.model.llama_model.generate(
                    inputs_embeds=n_embs,
                    max_new_tokens=max_new_tokens,
                    stopping_criteria=self.stopping_criteria,
                    num_beams=num_beams,
                    do_sample=do_sample,
                    min_length=min_length,
                    bad_words_ids=self.bad_words_ids,
                    max_length=max_length,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    temperature=temperature,
                )
            else:
                n_outputs = self.model.llama_model.generate(
                    inputs_embeds=n_embs,
                    max_new_tokens=max_new_tokens,
                    stopping_criteria=self.stopping_criteria,
                    num_beams=num_beams,
                    do_sample=do_sample,
                    min_length=min_length,
                    max_length=max_length,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    temperature=temperature,
                )

            n_output_token = n_outputs[0]
            if n_output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
                n_output_token = n_output_token[1:]
            if n_output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
                n_output_token = n_output_token[1:]
            n_output_text = self.model.llama_tokenizer.decode(n_output_token, add_special_tokens=False)
            n_output_text = n_output_text.split('###')[0]  # remove the stop sign '###'
            n_output_text = n_output_text.split('Assistant:')[-1].strip()
            conv.messages[-1][1] = n_output_text
        else:
            n_output_text = None

        q_current_max_len = q_embs.shape[1] + max_new_tokens
        if q_current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                  'The model will not see the contexts outside the range.')
        q_begin_idx = max(0, q_current_max_len - max_length)
        q_embs = q_embs[:, q_begin_idx:]
        if self.word_list is not None:
            q_outputs = self.model.llama_model.generate(
                inputs_embeds=q_embs,
                max_new_tokens=max_new_tokens,
                stopping_criteria=self.stopping_criteria,
                num_beams=num_beams,
                do_sample=do_sample,
                min_length=min_length,
                bad_words_ids=self.bad_words_ids,
                max_length=max_length,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                temperature=temperature,
            )
        else:
            q_outputs = self.model.llama_model.generate(
                inputs_embeds=q_embs,
                max_new_tokens=max_new_tokens,
                stopping_criteria=self.stopping_criteria,
                num_beams=num_beams,
                do_sample=do_sample,
                min_length=min_length,
                max_length=max_length,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                temperature=temperature,
            )

        q_output_token = q_outputs[0]
        if q_output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            q_output_token = q_output_token[1:]
        if q_output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            q_output_token = q_output_token[1:]
        q_output_text = self.model.llama_tokenizer.decode(q_output_token, add_special_tokens=False)
        q_output_text = q_output_text.split('###')[0]  # remove the stop sign '###'
        q_output_text = q_output_text.split('Assistant:')[-1].strip()

        return n_output_text, q_output_text #, n_output_token.cpu().numpy(), q_output_token.cpu().numpy()

    def upload_img(self, img1, img2, conv, img_list):
        if isinstance(img1, str):  # is a image path
            d_raw_image = Image.open(img1).convert('RGB')
            d_image = self.vis_processor(d_raw_image).unsqueeze(0).to(self.device)
        elif isinstance(img1, Image.Image):
            d_raw_image = img1
            d_image = self.vis_processor(d_raw_image).unsqueeze(0).to(self.device)
        elif isinstance(img1, torch.Tensor):
            if len(img1.shape) == 3:
                img1 = img1.unsqueeze(0)
            d_image = img1.to(self.device)
        n_image = None
        
        if isinstance(img2, str):  # is a image path
            q_raw_image = Image.open(img2).convert('RGB')
            q_image = self.vis_processor(q_raw_image).unsqueeze(0).to(self.device)
        elif isinstance(img2, Image.Image):
            q_raw_image = img2
            q_image = self.vis_processor(q_raw_image).unsqueeze(0).to(self.device)
        elif isinstance(img2, torch.Tensor):
            if len(img2.shape) == 3:
                img2 = img2.unsqueeze(0)
            q_image = img2.to(self.device)
        
        n_img_embeds, n_atts_img, q_img_embeds, q_atts_img = self.model.encode_img(d_image, n_image, q_image)

        img_list.append([n_img_embeds, q_img_embeds])
        conv.append_message(conv.roles[0], "<Img><ImageHere></Img>")
        msg = "Received."
        return msg

    def get_context_emb(self, conv, img_list):
        n_img_list = []
        q_img_list = []
        for img_embeds in img_list:
            n_img_list.append(img_embeds[0])
            q_img_list.append(img_embeds[1])

        prompt = conv.get_prompt()
        prompt_segs = prompt.split('<ImageHere>')
        assert len(prompt_segs) == len(n_img_list) + 1 == len(q_img_list) + 1, "Unmatched numbers of image placeholders and images."
        seg_tokens = [
            self.model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [self.model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        
        if n_img_list[0] is not None:
            n_mixed_embs = [emb for pair in zip(seg_embs[:-1], n_img_list) for emb in pair] + [seg_embs[-1]]
            n_mixed_embs = torch.cat(n_mixed_embs, dim=1)
        else:
            n_mixed_embs = None

        q_mixed_embs = [emb for pair in zip(seg_embs[:-1], q_img_list) for emb in pair] + [seg_embs[-1]]
        q_mixed_embs = torch.cat(q_mixed_embs, dim=1)
        return n_mixed_embs, q_mixed_embs