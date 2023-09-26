import logging
import random
import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
import torch.nn.functional as F

from vir_vlfm.common.registry import registry
from vir_vlfm.models.blip2 import Blip2Base, disabled_train
from vir_vlfm.models.modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer

class ViewpointRegistrationFlow(nn.Module):
    def __init__(self, inplane, outplane, hw_shape, kernel_size=3):
        super(ViewpointRegistrationFlow, self).__init__()
        self.down_1 = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.down_2 = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane*2, 4, kernel_size=kernel_size, padding=1, bias=False)
        self.hw_shape = hw_shape

    def forward(self, x1, x2):
        out1 = x1[:, 1:]
        out2 = x2[:, 1:]
        B, _, C = out1.shape
        out1_origin = out1.reshape(B, self.hw_shape[0], self.hw_shape[1],
                            C).permute(0, 3, 1, 2).contiguous()
        out2_origin = out2.reshape(B, self.hw_shape[0], self.hw_shape[1],
                            C).permute(0, 3, 1, 2).contiguous()

        out1 = self.down_1(out1_origin)
        out2 = self.down_2(out2_origin)
        flow = self.flow_make(torch.cat([out1, out2], 1))
        out1 = self.flow_warp(out1_origin, flow[:, :2], size=self.hw_shape)
        out2 = self.flow_warp(out2_origin, flow[:, 2:], size=self.hw_shape)

        add1 = out1_origin + out2
        add2 = out2_origin + out1
        out1 = add1.reshape(B, C, self.hw_shape[0] * self.hw_shape[1]).permute(0, 2, 1).contiguous()
        out1 = torch.cat([x1[:, :1], out1], 1)
        out2 = add2.reshape(B, C, self.hw_shape[0] * self.hw_shape[1]).permute(0, 2, 1).contiguous()
        out2 = torch.cat([x2[:, :1], out2], 1)
        return out1, out2

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()
        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm
        output = F.grid_sample(input, grid, align_corners=True)
        return output


@registry.register_model("vir_vlfm")
class VIR_VLFM(Blip2Base):
    """
    BLIP2 GPT-LLAMA model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/vir_vlfm.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        freeze_qformer=True,
        num_query_token=32,
        llama_model="",
        prompt_path="",
        prompt_template="",
        max_txt_len=32,
        end_sym='\n',
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
        use_adapter=False,
        scale=0.5
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource

        print('Loading VIT')
        self.vit_model = vit_model
        self.visual_encoder, self.ln_vision = self.init_vision_encoder_fuse(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision, use_adapter, scale
        )
        for n, m in self.visual_encoder.named_modules():
            if 'Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)

        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                if 'Adapter' in name:
                    print('DONT freeze:', name)
                    continue
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train

            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = disabled_train
            logging.info("freeze vision encoder")
            print("freeze vision encoder")
        print('Loading VIT Done')

        print('Loading Q-Former')

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
            )

        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.load_from_pretrained(url_or_filename=q_former_model)

        if freeze_qformer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            self.query_tokens.requires_grad = False
            logging.info("freeze Qformer")
            print("freeze Qformer")
        print('Loading Q-Former Done')

        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={'': device_8bit}
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
            )

        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        print('Loading LLAMA Done')

        self.llama_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llama_model.config.hidden_size
        )

        self.flow = ViewpointRegistrationFlow(self.visual_encoder.num_features, self.visual_encoder.num_features // 2, (16, 16))
        self.change_att = nn.Linear(
            2 * self.Qformer.config.hidden_size, self.Qformer.config.hidden_size
        )
        self.llama_proj_1 = nn.Linear(
            2 * self.Qformer.config.hidden_size, self.Qformer.config.hidden_size
        )

        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<ImageHere>" in raw_prompt]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []

    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()
    
    def semantic_emphasizing(self, input_1, input_2):
        input_before = torch.cat([input_1, input_2], 2)
        input_after = torch.cat([input_2, input_1], 2)
        att_weight_before = F.sigmoid(self.change_att(input_before))
        att_weight_after = F.sigmoid(self.change_att(input_after))
        att_1 = input_1 * att_weight_before
        att_2 = input_2 * att_weight_after
        return att_1, att_2

    def encode_img(self, d_image, n_image, q_image):
        device = d_image.device
        if self.low_resource:
            self.vit_to_cpu()
            d_image = d_image.to("cpu")
            if n_image is not None:
                n_image = n_image.to("cpu")
            q_image = q_image.to("cpu")

        with self.maybe_autocast():
            if n_image is not None:
                dn_image_embeds, n_image_embeds = self.visual_encoder(d_image, n_image)
                dn_image_embeds = dn_image_embeds.to(device)
                n_image_embeds = n_image_embeds.to(device)
            dq_image_embeds, q_image_embeds = self.visual_encoder(d_image, q_image)
            dq_image_embeds = dq_image_embeds.to(device)
            q_image_embeds = q_image_embeds.to(device)

            if n_image is not None:
                dn_image_embeds = self.ln_vision(dn_image_embeds)
                n_image_embeds = self.ln_vision(n_image_embeds)
            dq_image_embeds = self.ln_vision(dq_image_embeds)
            q_image_embeds = self.ln_vision(q_image_embeds)

            if n_image is not None:
                dn_image_embeds, n_image_embeds = self.flow(dn_image_embeds, n_image_embeds)
            dq_image_embeds, q_image_embeds = self.flow(dq_image_embeds, q_image_embeds)

            if n_image is not None:
                dn_image_atts = torch.ones(dn_image_embeds.size()[:-1], dtype=torch.long).to(device)
                dn_query_tokens = self.query_tokens.expand(dn_image_embeds.shape[0], -1, -1)
                dn_query_output = self.Qformer.bert(
                    query_embeds=dn_query_tokens,
                    encoder_hidden_states=dn_image_embeds,
                    encoder_attention_mask=dn_image_atts,
                    return_dict=True,
                )

            dq_image_atts = torch.ones(dq_image_embeds.size()[:-1], dtype=torch.long).to(device)
            dq_query_tokens = self.query_tokens.expand(dq_image_embeds.shape[0], -1, -1)
            dq_query_output = self.Qformer.bert(
                query_embeds=dq_query_tokens,
                encoder_hidden_states=dq_image_embeds,
                encoder_attention_mask=dq_image_atts,
                return_dict=True,
            )
            if n_image is not None:
                n_image_atts = torch.ones(n_image_embeds.size()[:-1], dtype=torch.long).to(device)
                n_query_tokens = self.query_tokens.expand(n_image_embeds.shape[0], -1, -1)
                n_query_output = self.Qformer.bert(
                    query_embeds=n_query_tokens,
                    encoder_hidden_states=n_image_embeds,
                    encoder_attention_mask=n_image_atts,
                    return_dict=True,
                )

            q_image_atts = torch.ones(q_image_embeds.size()[:-1], dtype=torch.long).to(device)
            q_query_tokens = self.query_tokens.expand(q_image_embeds.shape[0], -1, -1)
            q_query_output = self.Qformer.bert(
                query_embeds=q_query_tokens,
                encoder_hidden_states=q_image_embeds,
                encoder_attention_mask=q_image_atts,
                return_dict=True,
            )

            if n_image is not None:
                n_att_1, n_att_2 = self.semantic_emphasizing(dn_query_output.last_hidden_state, n_query_output.last_hidden_state)
                n_feats = torch.cat([n_att_1, n_att_2], 2)

            q_att_1, q_att_2 = self.semantic_emphasizing(dq_query_output.last_hidden_state, q_query_output.last_hidden_state)
            q_feats = torch.cat([q_att_1, q_att_2], 2)

            if n_image is not None:
                n_feats = self.llama_proj_1(n_feats)
                n_inputs_llama = self.llama_proj(n_feats)
                n_atts_llama = torch.ones(n_inputs_llama.size()[:-1], dtype=torch.long).to(n_image.device)

            q_feats = self.llama_proj_1(q_feats)
            q_inputs_llama = self.llama_proj(q_feats)
            q_atts_llama = torch.ones(q_inputs_llama.size()[:-1], dtype=torch.long).to(q_image.device)

        if n_image is not None:
            return n_inputs_llama, n_atts_llama, q_inputs_llama, q_atts_llama
        else:
            return None, None, q_inputs_llama, q_atts_llama

    def prompt_wrap(self, img_embeds, atts_img, prompt):
        if prompt:
            batch_size = img_embeds.shape[0]
            p_before, p_after = prompt.split('<ImageHere>')
            p_before_tokens = self.llama_tokenizer(
                p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_after_tokens = self.llama_tokenizer(
                p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
            wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
            wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
            return wrapped_img_embeds, wrapped_atts_img
        else:
            return img_embeds, atts_img

    def forward(self, samples):

        d_image = samples['d_image']
        if 'n_image' in samples.keys():
            n_image = samples['n_image']
        else:
            n_image = None
        q_image = samples['q_image']

        n_img_embeds, n_atts_img, q_img_embeds, q_atts_img = self.encode_img(d_image, n_image, q_image)
        if hasattr(samples, 'question_split'):  # VQA dataset
            print('VQA Batch')
            vqa_prompt = '###Human: <Img><ImageHere></Img> '
            if n_img_embeds is not None:
                n_img_embeds, n_atts_img = self.prompt_wrap(n_img_embeds, n_atts_img, vqa_prompt)
            q_img_embeds, q_atts_img = self.prompt_wrap(q_img_embeds, q_atts_img, vqa_prompt)
        elif self.prompt_list:
            prompt = random.choice(self.prompt_list)
            if n_img_embeds is not None:
                n_img_embeds, n_atts_img = self.prompt_wrap(n_img_embeds, n_atts_img, prompt)
            q_img_embeds, q_atts_img = self.prompt_wrap(q_img_embeds, q_atts_img, prompt)

        self.llama_tokenizer.padding_side = "right"

        if n_img_embeds is not None:
            n_text = [t + self.end_sym for t in samples["no_change_cap"]]
            n_to_regress_tokens = self.llama_tokenizer(
                n_text,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                add_special_tokens=False
            ).to(d_image.device)
            n_targets = n_to_regress_tokens.input_ids.masked_fill(
                n_to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
            )

        q_text = [t + self.end_sym for t in samples["change_cap"]]
        q_to_regress_tokens = self.llama_tokenizer(
            q_text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(d_image.device)
        q_targets = q_to_regress_tokens.input_ids.masked_fill(
            q_to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )

        empty_targets = (
            torch.ones([q_atts_img.shape[0], q_atts_img.shape[1]+1],
                       dtype=torch.long).to(d_image.device).fill_(-100)  # plus one for bos
        )

        if n_img_embeds is not None:
            n_targets = torch.cat([empty_targets, n_targets], dim=1)
        q_targets = torch.cat([empty_targets, q_targets], dim=1)

        batch_size = q_img_embeds.shape[0]
        if n_img_embeds is not None:
            n_bos = torch.ones([batch_size, 1],
                            dtype=n_to_regress_tokens.input_ids.dtype,
                            device=n_to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
            n_bos_embeds = self.llama_model.model.embed_tokens(n_bos)
            n_atts_bos = n_atts_img[:, :1]
            n_to_regress_embeds = self.llama_model.model.embed_tokens(n_to_regress_tokens.input_ids)
            n_inputs_embeds = torch.cat([n_bos_embeds, n_img_embeds, n_to_regress_embeds], dim=1)
            n_attention_mask = torch.cat([n_atts_bos, n_atts_img, n_to_regress_tokens.attention_mask], dim=1)

        q_bos = torch.ones([batch_size, 1],
                         dtype=q_to_regress_tokens.input_ids.dtype,
                         device=q_to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        q_bos_embeds = self.llama_model.model.embed_tokens(q_bos)
        q_atts_bos = q_atts_img[:, :1]
        q_to_regress_embeds = self.llama_model.model.embed_tokens(q_to_regress_tokens.input_ids)
        q_inputs_embeds = torch.cat([q_bos_embeds, q_img_embeds, q_to_regress_embeds], dim=1)
        q_attention_mask = torch.cat([q_atts_bos, q_atts_img, q_to_regress_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            if n_img_embeds is not None:
                n_outputs = self.llama_model(
                    inputs_embeds=n_inputs_embeds,
                    attention_mask=n_attention_mask,
                    return_dict=True,
                    labels=n_targets,
                )
            q_outputs = self.llama_model(
                inputs_embeds=q_inputs_embeds,
                attention_mask=q_attention_mask,
                return_dict=True,
                labels=q_targets,
            )
        if n_img_embeds is not None:
            loss = n_outputs.loss + q_outputs.loss
        else:
            loss = q_outputs.loss
        return {"loss": loss}

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        q_former_model = cfg.get("q_former_model", "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llama_model = cfg.get("llama_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        freeze_qformer = cfg.get("freeze_qformer", True)
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)

        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')

        use_adapter = cfg.get("use_adapter", False)
        scale = cfg.get("scale", 0.5)

        model = cls(
            vit_model=vit_model,
            q_former_model=q_former_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            freeze_qformer=freeze_qformer,
            num_query_token=num_query_token,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            use_adapter=use_adapter,
            scale=scale,
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights
        if ckpt_path:
            print("Load BLIP2-LLM Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            new_ckpt = {}
            for k, v in ckpt['model'].items():
                if 'llm_' in k:
                    k_new = k.replace('llm_', 'llama_')
                    print(k, 'is changed to ', k_new)
                else:
                    k_new = k
                new_ckpt[k_new] = v
            msg = model.load_state_dict(new_ckpt, strict=False)
            print('unexpected_keys:', msg[1])

        return model
