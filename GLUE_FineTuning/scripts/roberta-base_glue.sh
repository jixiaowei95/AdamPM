export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="Data/GLUE"  # Create your cache directory
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0

accelerate launch --num_processes 1 --num_machines 1 --mixed_precision bf16 run_glue.py \
    --model_name_or_path roberta-base \
    --task_name mrpc \
    --max_length 512 \
    --seed 1234 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    \
    --optimizer adampm \
    --lr 2e-5 \
    --beta1 0.9 \
    --beta2 0.95 \
    --weight_decay 0.0 \
    --group_size 2 \
    \
    --num_train_epochs 30 \
    --output_dir scripts/mrpc_A \
    --mnli_output_dir scripts/mnli_B \
    \
    