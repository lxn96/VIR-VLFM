## How to Prepare Vicuna Weight
Vicuna is an open-source LLAMA-based LLM that has a performance close to ChatGPT. 
We currently use the v0 version of Vicuna-7B. 

To prepare Vicuna’s weight, first download Vicuna’s **delta** weight from [https://huggingface.co/lmsys/vicuna-7b-delta-v0](https://huggingface.co/lmsys/vicuna-7b-delta-v0). 
In case you have git-lfs installed (https://git-lfs.com), this can be done by

```
git lfs install
git clone https://huggingface.co/lmsys/vicuna-7b-delta-v0
```


Then, you need to obtain the original LLAMA-7B weights in the HuggingFace format 
either following the instruction provided by HuggingFace 
[here](https://huggingface.co/docs/transformers/main/model_doc/llama) or from the Internet. 

When these two weights are ready, we can use tools from Vicuna’s team to create the real working weight.
First, Install their library that is compatible with v0 Vicuna by

```
pip install git+https://github.com/lm-sys/FastChat.git@v0.1.10
```

Then, run the following command to create the final working weight

```
python -m fastchat.model.apply_delta --base /path/to/llama-7bOR7b-hf/  --target /path/to/save/working/vicuna/weight/  --delta /path/to/vicuna-7bOR7b-delta-v0/
```

Now you are good to go!

