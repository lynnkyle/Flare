import os
import argparse
import numpy as np
import torch.cuda

from dataset import make_data_module
from model import EmbeddingModel, KGELlama
from utils import get_logger, get_accelerate_model, SavePeftModelCallback
from transformers import HfArgumentParser, set_seed, GenerationConfig, AutoTokenizer, AutoConfig, LlamaForCausalLM, \
    Seq2SeqTrainer
from utils import ModelArguments, DataArguments, TrainingArguments, GenerationArguments


# train训练过程
def train():
    hf_parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, GenerationArguments))
    model_args, data_args, training_args, generation_args, _ = hf_parser.parse_args_into_dataclasses(
        return_remaining_strings=True)
    training_args.generation_config = GenerationConfig(**vars(generation_args))
    args = argparse.Namespace(**vars(model_args), **vars(data_args), **vars(training_args))
    assert args.model_class in ['KGELlama', 'KGEBert']
    if args.kge_model == 'TransE':
        args.embedding_dim = 250
    set_seed(args.seed)

    os.makedirs(args.output_dir)
    logger = get_logger(args.output_dir)
    logger.info(vars(args))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    if args.model_class == 'KGELlama':
        tokenizer.add_tokens(['[QUERY]', '[ENTITY]'])

    model_config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = get_accelerate_model(args, model_config, LlamaForCausalLM)
    model.config.use_cache = False

    if args.model_class == 'KGELlama':
        llm_config = model.config
        kge_embedding_dir = args.dataset
        embed_model = EmbeddingModel(kge_embedding_dir, args.embedding_dim, 1024, llm_config.hidden_size,
                                     llm_config.hidden_act)
        model = KGELlama(tokenizer, model, embed_model)

    data_module = make_data_module(args, tokenizer, logger)

    trainer = Seq2SeqTrainer(
        model=model,
        # tokenizer=tokenizer,
        args=training_args,
        **data_module
    )
    if not args.full_finetune:
        trainer.add_callback(SavePeftModelCallback)  # 训练完成后，只保存PEFT adapter权重，而不是整个模型

    if args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.save_metrics('train', metrics)  # 保存训练过程中的评估指标(损失、Hit1...)
        trainer.save_state()  # 保存训练过程的中间状态(优化器状态、学习率调度器状态、当前的epoch, step)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    os.environ['NCCL_P2P_DISABLE'] = '1'
    os.environ['NCCL_IB_DISABLE'] = '1'
    torch.cuda.set_device(0)
    train()
