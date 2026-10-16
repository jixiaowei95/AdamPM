export CUDA_VISIBLE_DEVICES=0

torchrun --standalone --nproc_per_node 1 torchrun_main.py \
    --model_config configs/llama_130m.json \
    --batch_size 64 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 2000 \
    --dtype bfloat16 \
    --dataset_path c4/en \
    --eval_every 1000 \
    --save_every 9999999 \
    \
    --optimizer adampm \
    --lr 0.002 \
    --beta1 0.9 \
    --beta2 0.95 \
    --weight_decay 0.0 \
    --group_size 2 \
    \
    --compile \
    --compile_mode default \
    
