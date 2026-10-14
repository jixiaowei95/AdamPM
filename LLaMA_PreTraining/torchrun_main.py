import os
import time
import json
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist

import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM as HF_LlamaForCausalLM
from transformers.utils import cached_file

import datasets
import datasets.distributed

from tqdm import tqdm
from loguru import logger

from peft_pretraining import training_utils, args_utils
from peft_pretraining.dataloader import PreprocessedIterableDataset
from peft_pretraining.modeling_llama import LlamaForCausalLM
from args_parser import parse_args

import bitsandbytes as bnb

import itertools

from optimizer_torch import *
from MemoryMeasure.gpu_mem_utils import *


transformers.logging.set_verbosity_error()





@torch.no_grad()
def evaluate_model(model, preprocess_batched, pad_idx, global_rank, world_size, device, batch_size):
    _time = time.time()

    """ here is load c4 dataset (validation dataset) """
    if args.dataset_path is not None:
        logger.info(f"Using local validation dataset for validation")
        data_files_val= {"validation": [f"{args.dataset_path}/c4-validation.{str(i).zfill(5)}-of-00008.json.gz" for i in range(0,8)]}
        val_data = datasets.load_dataset(path=args.dataset_path,  data_files=data_files_val, split="validation", streaming=True)
    else:
        print("Load the C4 dataset (validation dataset) in streaming mode : allenai/c4, en, validation, streaming=True")
        val_data = datasets.load_dataset("allenai/c4", "en", split="validation", streaming=True)  # DGX

    val_data = val_data.shuffle(seed=42)
    logger.info(f"Loaded validation dataset in {time.time() - _time:.2f} seconds")

    if not args.single_gpu:
        val_data = datasets.distributed.split_dataset_by_node(val_data, rank=global_rank, world_size=world_size)

    val_data_mapped = val_data.map(
        preprocess_batched,
        batched=True,
        remove_columns=["text", "timestamp", "url"],
    )
    val_data_mapped.batch = lambda batch_size: training_utils.batch_fn(val_data_mapped, batch_size)

    target_eval_tokens = 10_000_000
    
    evaluated_on_tokens = 0
    total_loss = torch.tensor(0.0).to(device)
    total_batches = 1
    logger.info(f"Eval set prepared in {time.time() - _time:.2f} seconds")

    for batch in val_data_mapped.batch(batch_size=batch_size):
        if evaluated_on_tokens > target_eval_tokens:
            break
        total_batches += 1

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        loss = model(**batch, labels=labels).loss
        total_loss += loss.detach()

        evaluated_on_tokens += (batch["input_ids"] != pad_idx).sum().item() * world_size

    total_loss = total_loss / total_batches

    # Gather losses across all GPUs
    gathered_losses = [torch.zeros_like(total_loss) for _ in range(world_size)]
    dist.all_gather(gathered_losses, total_loss)
    total_loss = sum([t.item() for t in gathered_losses]) / world_size

    return total_loss, evaluated_on_tokens



def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    assert "LOCAL_RANK" in os.environ, "torchrun should set LOCAL_RANK"
    global_rank = int(os.environ['RANK'])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    
    logger.info(f"Global rank {global_rank}, local rank {local_rank}, device: {torch.cuda.current_device()}")

    dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size)

    logger.info("Process group initialized")
    device = f"cuda:{local_rank}"

    if args.total_batch_size is not None:
        if args.gradient_accumulation is None:
            assert args.total_batch_size % world_size == 0, "total_batch_size must be divisible by world_size"
            args.gradient_accumulation = args.total_batch_size // (args.batch_size * world_size)
            assert args.gradient_accumulation > 0, "gradient_accumulation must be greater than 0"

    assert args.gradient_accumulation * args.batch_size * world_size == args.total_batch_size, \
        "gradient_accumulation * batch_size * world_size must be equal to total_batch_size"

    # turn off logger
    if global_rank != 0: logger.remove()

    
    logger.info(f"Using dist with rank {global_rank} (only rank 0 will log)")
    logger.info("*" * 40)
    logger.info(f"Starting training with the arguments")
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)

    logger.info(f"\nargs.scheduler = {args.scheduler}\n")


    # it doesn't matter which tokenizer we use, because we train from scratch
    # T5 tokenizer was trained on C4 and we are also training on C4, so it's a good choice
    try:
        tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=args.max_length, local_files_only=True)
        tokenizer_file = cached_file("t5-base", "tokenizer_config.json", local_files_only=True)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=args.max_length)


    def preprocess_batched(batch):
        batch = tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return batch

    model_config = AutoConfig.from_pretrained(args.model_config)
    if args.use_hf_model:
        model: HF_LlamaForCausalLM = AutoModelForCausalLM.from_config(model_config)
    else:
        model = LlamaForCausalLM(model_config)

    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()

    global_step = 0
    update_step = 0
    beginning_step = 0
    tokens_seen = 0
    tokens_seen_before = 0
    
    loss_accum = 0

    if args.continue_from is not None:
        logger.info("*" * 40)
        logger.info(f"Loading model from {args.continue_from}")
        checkpoint_path = os.path.join(args.continue_from, "pytorch_model.bin")
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)
        logger.info(f"Model successfully loaded (strict=True policy)")

        if os.path.exists(os.path.join(args.continue_from, "training_state.json")):
            logger.info(f"Loading training state like global_step, update_step, and tokens_seen from {args.continue_from}")
            with open(os.path.join(args.continue_from, "training_state.json")) as f:
                _old_state = json.load(f)
            global_step = _old_state["global_step"]
            update_step = _old_state["update_step"]
            tokens_seen = _old_state["tokens_seen"]
            tokens_seen_before = _old_state["tokens_seen_before"]
            logger.info(f"global_step       : {global_step}")
            logger.info(f"update_step       : {update_step}")
            logger.info(f"tokens_seen       : {tokens_seen}")
            logger.info(f"tokens_seen_before: {tokens_seen_before}")
            logger.info(f"Will train for {args.num_training_steps - update_step} update steps")
        else:
            logger.warning(f"Did not find training state in {args.continue_from}, global step will start from zero")
        logger.info("*" * 40)


    if args.dtype in ["bf16", "bfloat16"]:
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device)


    # ======================  torch.compile  ============
    if args.compile:
        torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
        if torch_version < (2, 0):
            logger.error(f"torch.compile requires PyTorch 2.0 or higher (current: {torch.__version__}), disabling compile")
        else:
            logger.info(f"Compiling model with torch.compile: "
                        f"mode={args.compile_mode}, dynamic={args.compile_dynamic}, dtype={args.dtype}")
            model = torch.compile(
                model,
                mode=args.compile_mode,
                dynamic=args.compile_dynamic,
                # fullgraph=True,
            )
    # ======================================================


    n_total_params = sum(p.numel() for p in model.parameters())
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    # 
    run_config = dict(vars(args))
    run_config.update({
        "max_lr": run_config.pop("lr"),  # rename lr to max_lr to avoid conflicts with scheduler
        "total_params_M": n_total_params / 1_000_000,
        "dataset": 'c4',
        "model": model_config.to_dict(),
        "world_size": world_size,
        "device": str(device),
    })

    if global_rank == 0:
        # fix tqdm visual length to 80 so that the progress bar
        # doesn't jump around when changing from external display to laptop
        pbar = tqdm(total=args.num_training_steps - update_step, desc="Update steps", ncols=80)




        
        




    # Print out the names and shapes of all the weight parameters of the network
    for name, param in model.named_parameters():
        print(f"{name:70s} , shape={tuple(param.shape)} , requires_grad={param.requires_grad}")


    # print params and trainable params
    logger.info(f"\nmodel = {model}\n")
    logger.info(f"Total params: {sum(p.numel() for p in model.parameters()) / 1_000_000:.6f} M")
    logger.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.6f} M")
    logger.info(f"Memory of weights : {sum(p.numel() * p.element_size() for p in model.parameters())} bytes")
    logger.info(f"Memory of weights : {sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2):.4f} MiB\n")



    layer_wise_flag = False
    print(f"\nargs.optimizer.lower() = {args.optimizer.lower()} \n")
    if args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(trainable_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)

    elif args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(trainable_params, lr=args.lr, weight_decay=args.weight_decay, momentum=args.beta1)

    elif args.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, betas=(args.beta1,args.beta2), weight_decay=args.weight_decay)
        
    elif args.optimizer.lower() == "adam_mini":
        optimizer = Adam_mini(model.named_parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay,
                              dim=model_config.hidden_size, n_heads=model_config.num_attention_heads)

    elif args.optimizer.lower() == "adampm":
        optimizer = AdamPM(trainable_params, lr=args.lr, betas=(args.beta1,args.beta2), weight_decay=args.weight_decay, group_size=args.group_size)
        
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")
    
    print(f"\n\noptimizer = {optimizer} \n")




    if not layer_wise_flag:
        scheduler = training_utils.get_scheculer(
            optimizer=optimizer,
            scheduler_type=args.scheduler,
            num_training_steps=args.num_training_steps,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
        )



    if not args.single_gpu:
        model: LlamaForCausalLM = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )

    # global steps and others are defined above
    pad_idx = tokenizer.pad_token_id
    update_time = time.time()
    local_step = 0  # when continue_from is used, local_step != global_step




    # ##############################
    # TRAINING LOOP with multiple epochs
    # ##############################

    for epoch in range(args.num_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{args.num_epochs}")
        
        """ here is load c4 dataset (training dataset) - Reload the complete dataset at the end of each epoch """
        if args.dataset_path is not None:
            logger.info(f"Using local training dataset for training, epoch {epoch + 1}\n")
            data_files_train = {"train": [f"{args.dataset_path}/c4-train.{str(i).zfill(5)}-of-01024.json.gz" for i in range(0,1024)]}
            data = datasets.load_dataset(path=args.dataset_path,  data_files=data_files_train, split="train", streaming=True)
            logger.info(f"loaded dataset for epoch {epoch + 1}")
        else:
            print("Load the C4 dataset (training dataset) in streaming mode : allenai/c4, en, train, streaming=True")
            data = datasets.load_dataset("allenai/c4", "en", split="train", streaming=True)  # DGX

        seed_for_shuffle = 42 + epoch
        logger.info(f"Shuffling data with seed {seed_for_shuffle} for epoch {epoch + 1}")
        data: datasets.Dataset = data.shuffle(seed=seed_for_shuffle)
        if not args.single_gpu:
            data = datasets.distributed.split_dataset_by_node(
                data, rank=global_rank, world_size=world_size,
            )

        dataset = PreprocessedIterableDataset(data, tokenizer, batch_size=args.batch_size, max_length=args.max_length)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=args.workers)

        for batch_idx, batch in enumerate(dataloader):

            global_step += 1
            local_step += 1

            if update_step > args.num_training_steps:
                logger.info(f"Reached max number of update steps (f{args.num_training_steps}). Stopping training.")
                print(f"Rank {global_rank} stopping training.")
                break

            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["input_ids"].clone()
            labels[labels == pad_idx] = -100
            tokens_seen += (batch["input_ids"] != pad_idx).sum().item() * world_size

            loss = model(**batch, labels=labels).loss
            scaled_loss = loss / args.gradient_accumulation
            loss_accum += scaled_loss.detach()
            scaled_loss.backward()

            if global_step % args.gradient_accumulation != 0:
                continue


            # The below code is only executed during the update step
            
            # add grad clipping
            if args.grad_clipping != 0.0: torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clipping)

            if global_rank == 0: 
                pbar.set_postfix(loss=f"{loss_accum.item():.4f}") 
                pbar.update(1)
            
            if not layer_wise_flag:
                if global_rank == 0 and (update_step == 0 or update_step == 8):
                    torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.synchronize(device)
                
                optimizer.step()
                
                if global_rank == 0 and (update_step == 0 or update_step == 8):
                    torch.cuda.synchronize(device)
                
                scheduler.step()
                optimizer.zero_grad()

            update_step += 1
            update_time = time.time() - update_time



            if global_rank == 0 and not layer_wise_flag:
                if update_step == 7 or update_step == 950:
                    print(f"\n===== Step {update_step} : print GPU Memory =====")
                    print_optimizer_state_breakdown(optimizer, prefix="  ")
                    compute_and_print_weight_memory(model, prefix="  ")
                    print_cuda_memory_breakdown(model, optimizer, prefix="  ")
                elif update_step % 1000 == 0:
                    v_bytes = optimizer_state_v_bytes(optimizer)
                    m_bytes = optimizer_state_m_bytes(optimizer)
                    print(f"\nStep : {update_step}, Second Moment : {v_bytes / (1024**2):.3f} MiB , First Moment = {m_bytes / (1024**2):.3f} MiB")



            # save checkpoint by save_every
            if local_step > args.gradient_accumulation and update_step % args.save_every == 0 and global_rank == 0:
                print(f"\nargs.save_every = {args.save_every} \n")
                current_model_directory = f"{args.save_dir}/model_{update_step}"
                logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
                os.makedirs(args.save_dir, exist_ok=True)
                
                actual_model = model.module if hasattr(model, 'module') else model

                # pad_token_id
                if actual_model.config.pad_token_id is None or actual_model.config.pad_token_id < 0:
                    actual_model.config.pad_token_id = tokenizer.pad_token_id
                if hasattr(actual_model, 'generation_config') and (actual_model.generation_config is None or actual_model.generation_config.pad_token_id is None or actual_model.generation_config.pad_token_id < 0):
                    if not hasattr(actual_model, 'generation_config'):
                        from transformers import GenerationConfig
                        actual_model.generation_config = GenerationConfig()
                    actual_model.generation_config.pad_token_id = tokenizer.pad_token_id

                print(f"\ncurrent_model_directory = {current_model_directory}\n")

                actual_model.save_pretrained(current_model_directory, max_shard_size='100GB')



                optimizer_checkpoint = {
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "update_step": update_step,
                    "global_step": global_step,
                    "config": run_config,
                    "dtype": args.dtype,
                }
                torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

                training_state_checkpoint = {
                    "global_step": global_step,
                    "update_step": update_step,
                    "tokens_seen": tokens_seen,
                    "tokens_seen_before": tokens_seen_before,
                    "update_time": update_time,
                }
                with open(f"{current_model_directory}/training_state.json", "w") as f:
                    json.dump(training_state_checkpoint, f, indent=4)


            # evaluation
            if update_step % args.eval_every == 0:
                logger.info(f"Performing evaluation at step {update_step}")
                total_loss, evaluated_on_tokens = evaluate_model(
                    model, preprocess_batched, pad_idx, global_rank, world_size, device, args.batch_size
                )
                logger.info(f"Eval loss and perplexity at step {update_step}: {np.exp(total_loss)}")

                # =================== Current learning rate ===================
                if not layer_wise_flag:
                    current_lr = optimizer.param_groups[0]["lr"]
                else:
                    current_lr = list(optimizer_dict.values())[0].param_groups[0]["lr"]
                logger.info(f"Current learning rate at step {update_step}: {current_lr}")
                # =================== Current learning rate ===================


            if not layer_wise_flag:
                lr = optimizer.param_groups[0]["lr"]
            else:
                lr = list(optimizer_dict.values())[0].param_groups[0]["lr"]
            tokens_in_update = tokens_seen - tokens_seen_before
            tokens_seen_before = tokens_seen
            batches_in_update = args.gradient_accumulation * world_size

            loss_accum = 0

            update_time = time.time()

        if update_step > args.num_training_steps:
            print(f"\nupdate_step {update_step} > args.num_training_steps {args.num_training_steps} \n")
            break

    # ##############################
    # END of training loop
    # ##############################
    
    
    
    logger.info("Training finished")
    if global_rank == 0: pbar.close()

    # Final evaluation
    logger.info("Running final evaluation")
    model.eval()
    del loss, optimizer, scheduler
    import gc; gc.collect()
    torch.cuda.empty_cache()

    total_loss, evaluated_on_tokens = evaluate_model(
        model,
        preprocess_batched,
        pad_idx,
        global_rank,
        world_size,
        device,
        args.batch_size,
    )

    logger.info(f"Final Eval loss and perplexity at step {update_step}: {total_loss}, {np.exp(total_loss)}")

    logger.info("Script finished successfully")
    print(f"Rank {global_rank} finished successfully")


if __name__ == "__main__":
    print("######### Starting script ##########")
    args = parse_args(None)
    main(args)
    