import argparse
import torch
from peft_pretraining import training_utils, args_utils


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs to train for")
    
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["linear", "cosine", "cosine_restarts"])
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--use_hf_model", default=False, action="store_true")
    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--total_batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--activation_checkpointing", action="store_true", default=False)
    parser.add_argument("--warmup_steps", type=int, default=1_000)
    parser.add_argument("--eval_every", type=int, default=5_000)
    parser.add_argument("--num_training_steps", type=int, default=10_000,
                        help="Number of **update steps** to train for. "
                             "Notice that gradient accumulation is taken into account.")
    parser.add_argument("--max_train_tokens", type=training_utils.max_train_tokens_to_number, default=None,
                        help="Number of tokens to train on. Overwrites num_training_steps. "
                             "You can use M and B suffixes, e.g. 100M or 1B.")
    parser.add_argument("--save_every", type=int, default=10_000)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--tags", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_bf16_supported() else "float32")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--single_gpu", default=False, action="store_true")
    
    # AdamW parameters
    parser.add_argument("--optimizer", default="Adam")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clipping", type=float, default=0.0)
    
    # AdamPM parameters
    parser.add_argument("--group_size", type=int, default=2)


    # ======================  torch.compile  ======================
    parser.add_argument("--compile", action="store_true", default=False,
                        help="Enable torch.compile to accelerate model training/inference (need PyTorch >= 2.0)")
    parser.add_argument("--compile_mode", type=str, default="default",
                        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
                        help="torch.compile mode:\n"
                             "- default\n"
                             "- reduce-overhead\n"
                             "- max-autotune\n"
                             "- max-autotune-no-cudagraphs")
    parser.add_argument("--compile_dynamic", action="store_true", default=False,
                        help="Enable dynamic shape support (activated when the input sequence length / batch size is not fixed)")
    # ==============================================================


    args = parser.parse_args(args)
    args = args_utils.check_args_torchrun_main(args)
    return args

