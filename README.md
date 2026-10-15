# AdamPM: Memory-Efficient LLM Optimization via Moment Pooling



# 1. LLaMA Pretraining

## Prepare the data

Download the English C4 training and validation shards and  update `--dataset_path` in the scripts.

## Run pre-training

Run commands as follows. Before running, edit `CUDA_VISIBLE_DEVICES` and `--nproc_per_node` in the chosen script to match the available GPUs.

```bash
cd AdamPM/LLaMA_PreTraining
bash scripts/llama_60m.sh
```



# 2. GLUE Fine-Tuning



## Run

```bash
cd AdamPM/GLUE_FineTuning
bash scripts/roberta-base_glue.sh
```



Before the first run, disable Hugging Face Hub offline mode `HF_HUB_OFFLINE` and `roberta-base_glue.sh` will automatically download and cache:
- The pretrained RoBERTa-base model weights, configuration, and tokenizer.
- The dataset for the selected GLUE task.
- The corresponding GLUE evaluation metric module.



# 3. TinyLLaVA Supervised Fine-Tuning



This experiment is based on [TinyLLaVA Factory](https://github.com/TinyLLaVA/TinyLLaVA_Factory).



## Run

```bash
cd AdamPM/TinyLLaVA_Factory
bash scripts/train/train_qwen2_05b_base_adampm.sh
```

## Prepare the datasets

Before running `scripts/train/train_qwen2_05b_base_adampm.sh`, download the LLaVA-1.5 665K visual instruction-tuning mixture.
The script uses the following annotation file:

```text
/home/ai/data/llava/dataset/text_files/llava_v1_5_mix665k.json
```

The corresponding image root is:

```text
/home/ai/data/llava/dataset
```



## Prepare the pretrained TinyLLaVA checkpoint

Before supervised fine-tuning, a local stage-1 pretrained TinyLLaVA checkpoint must be prepared according to the TinyLLaVA Factory documentation.
For this experiment, the required checkpoint is:

```text
/mnt/data/sata/yinghu/checkpoints/llava_factory/tiny-llava-Qwen2-0.5B-siglip-so400m-patch14-384-base-pretrain
```





## Contact

Email: by2442103@buaa.edu.cn
