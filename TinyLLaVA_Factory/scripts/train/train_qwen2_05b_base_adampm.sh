#!/bin/bash

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." || exit 1

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1

FINETUNE_DATA_PATH=/home/ai/data/llava/dataset/text_files/llava_v1_5_mix665k.json
FINETUNE_IMAGE_PATH=/home/ai/data/llava/dataset


LLM_VERSION=Qwen/Qwen2-0.5B
VT_VERSION=google/siglip-so400m-patch14-384
VT_VERSION2=""
CN_VERSION=mlp2x_gelu
CONV_VERSION=qwen2_base
VERSION=base
TRAIN_RECIPE=common
MODEL_MAX_LENGTH=2048


PRETRAIN_DIR=/mnt/data/sata/yinghu/checkpoints/llava_factory/tiny-llava-Qwen2-0.5B-siglip-so400m-patch14-384-base-pretrain

for sub in language_model vision_tower connector; do
    if [ ! -f "$PRETRAIN_DIR/$sub/pytorch_model.bin" ]; then
        echo "ERROR: Lack Weights: $PRETRAIN_DIR/$sub/pytorch_model.bin"
        exit 1
    fi
done
echo "Pretraining checkpoint verified successfully: $PRETRAIN_DIR"

# ===== finetuning =====
bash scripts/train/finetune_ddp_adampm.sh "$FINETUNE_DATA_PATH" "$FINETUNE_IMAGE_PATH" "$LLM_VERSION" "$VT_VERSION" "$VT_VERSION2" "$CN_VERSION" "$CONV_VERSION" "$VERSION" "$TRAIN_RECIPE" "$MODEL_MAX_LENGTH"


