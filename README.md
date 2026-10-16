# AdamPM: Memory-Efficient LLM Optimization via Moment Pooling



# 1. LLaMA Pretraining

## Prepare the datasets

Download the training and validation shards of the cleaned English C4 dataset, then set `--dataset_path` in the training scripts (e.g., `scripts/llama_60m.sh`) to the directory containing the downloaded shards.

## Requirements

The required Python dependencies include `torch`, `transformers`, `datasets`, `numpy`, and other packages listed in `requirements.txt`.

## Run pre-training

Run the commands as follows, 

```bash
cd AdamPM/LLaMA_PreTraining
bash scripts/llama_60m.sh
```

Set `CUDA_VISIBLE_DEVICES` in the script to select the GPUs to use, and set `--nproc_per_node` to the number of selected GPUs (one process per GPU). Set `--optimizer` to select the optimizer and `--eval_every` to specify the number of training steps between evaluations, at which evaluation perplexity is reported.



# 2. GLUE Fine-Tuning

## Run GLUE fine-tuning

```bash
cd AdamPM/GLUE_FineTuning
bash scripts/roberta-base_glue.sh
```

Before the first run, disable Hugging Face Hub offline mode by commenting out `export HF_HUB_OFFLINE=1` in `roberta-base_glue.sh`. Running this  script will automatically download and cache any missing resources under the directory specified by `HF_HOME`:

- Model: The pretrained RoBERTa-base weights, model configuration, and tokenizer files.
- Dataset: The selected GLUE task’s dataset.
- Evaluation metric: The shared GLUE evaluation module, configured for the selected task.



Use `--task_name` to select the GLUE task and `--metric_for_best_model` to specify the metric used for that task.

The task-to-metric mapping is as follows:

| Task   | `--metric_for_best_model` |
| ------ | ------------------------- |
| `cola` | `matthews_correlation`    |
| `mrpc` | `f1`                      |
| `stsb` | `pearson`                 |
| `rte`  | `accuracy`                |
| `sst2` | `accuracy`                |
| `qqp`  | `accuracy`                |
| `mnli` | `accuracy`                |
| `qnli` | `accuracy`                |





# 3. Supervised Fine-Tuning of TinyLLaVA

This experiment is based on [TinyLLaVA Factory](https://github.com/TinyLLaVA/TinyLLaVA_Factory).

## Prepare the datasets

Before running the script, download the LLaVA-1.5 665K visual instruction-tuning mixture.
Update the dataset paths in `scripts/train/train_qwen2_05b_base_adampm.sh`:

```bash
FINETUNE_DATA_PATH=/path/to/llava_v1_5_mix665k.json  # Path to the annotation file
FINETUNE_IMAGE_PATH=/path/to/image/root  # Root directory containing the images
```

## Prepare the pretrained TinyLLaVA checkpoint

Before supervised fine-tuning, prepare a local pretrained TinyLLaVA checkpoint according to the TinyLLaVA Factory documentation.
Set `PRETRAIN_DIR` to the directory of the pretrained checkpoint:

```bash
PRETRAIN_DIR=/path/to/pretrained-tinyllava-checkpoint
```

## Run fine-tuning of TinyLLaVA

```bash
cd AdamPM/TinyLLaVA_Factory
bash scripts/train/train_qwen2_05b_base_adampm.sh
```



## Contact

Email: by2442103@buaa.edu.cn
