export CUDA_VISIBLE_DEVICES=0,1

torchrun --standalone --nproc_per_node 2 torchrun_main.py \
    --model_config configs/llama_350m.json \
    --batch_size 32 \
    --total_batch_size 512 \
    --num_training_steps 60000 \
    --warmup_steps 6000 \
    --dtype bfloat16 \
    --dataset_path c4/en \
    --eval_every 1000 \
    --save_every 9999999 \
    \
    --optimizer adampm \
    --lr 0.0006 \
    --beta1 0.9 \
    --beta2 0.95 \
    --weight_decay 0.0 \
    --group_size 2 \
    \
    --compile \
    --compile_mode default \
    
