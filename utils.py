import numpy as np
import os
import logging
from typing import Optional
from dataclasses import dataclass, field
import torch
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig
from peft.tuners.lora import LoraLayer
import transformers
from transformers import BitsAndBytesConfig, Seq2SeqTrainingArguments


def calculate_rank(score, target, filter_list):
    score = score.copy()
    target_score = score[target]
    score[filter_list] = target_score - 1
    score[target] = target_score
    rank = np.sum(score > target_score) + 1
    # rank = np.sum(score > target_score) + np.sum(score == target_score) // 2 + 1
    return rank


def metrics(rank_list):
    mr = np.mean(rank_list)
    mrr = np.mean(1 / rank_list)
    hit10 = np.sum(rank_list < 11) / len(rank_list)
    hit3 = np.sum(rank_list < 4) / len(rank_list)
    hit1 = np.sum(rank_list < 2) / len(rank_list)
    return mr, mrr, hit10, hit3, hit1


def get_rank(score, ent, filter_list):
    score = score.copy()
    return calculate_rank(score, ent, filter_list)


def get_topK(score, ent, filter_list, topK):
    score = score.copy()
    target_score = score[ent]
    score[filter_list] = target_score - 1
    score[ent] = target_score
    indices = np.argsort(-score, kind='stable')  # 负号表示降序
    vals = score[indices]
    return indices[:topK], vals[:topK]


@dataclass
class ModelArguments:
    model_class: str = field(
        default='KGELlama',
        metadata={'help': 'KGELLama(decoder-only) | KGEBert'}
    )
    model_name_or_path: Optional[str] = field(
        default='models--TheBloke--Llama-2-7B-fp16',
        metadata={'help': 'LLM Path'}
    )
    kge_model: Optional[str] = field(
        default='MyGo',
        metadata={'help': 'TransE | CoLE | SimKGC'}
    )
    embedding_dim: int = field(
        default=256,
        metadata={'help': 'Embedding Dimension For KGEModel'}
    )


@dataclass
class DataArguments:
    dataset: str = field(default="data/DB15K", metadata={'help': 'Fine Turn On Which Dataset'})
    train_path: str = field(default="data/DB15K/train.json", metadata={'help': 'Path For Train File'})
    eval_path: str = field(default="data/DB15K/valid.json", metadata={'help': 'Path For Valid File'})
    test_path: str = field(default="data/DB15K/test.json", metadata={'help': 'Path For Test File'})
    source_max_len: int = field(default=2048, metadata={'help': 'Maximum Source Sequence Length'})
    target_max_len: int = field(default=64, metadata={'help': 'Maximum Target Sequence Length'})


@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    # 大模型参数 微调与量化(高精度参数转换为低精度参数 float32-int4)
    full_finetune: bool = field(default=False, metadata={'help': 'FineTurn Entire Model Without Adapter'})
    use_quant: bool = field(default=False, metadata={'help': 'Use Quantized Model During Training Or Not'})
    double_quant: bool = field(default=True, metadata={'help': 'Compress Statistics Through Double Quantization.'})
    quant_type: str = field(default='nf4', metadata={'help': 'Quantization Type To Use fp4 Or nf4'})
    bits: int = field(default=4, metadata={'help': 'Bit Of Compress Statistics'})

    # KGEModel参数
    do_train: bool = field(default=True, metadata={'help': 'Train Or Not'})
    do_eval: bool = field(default=True, metadata={'help': 'Eval Or Not'})
    output_dir: str = field(default='./output', metadata={'help': 'Output Dir For Logs And Checkpoints'})

    dataloader_num_workers: int = field(default=8, metadata={'help': 'Treads To Load Data'})
    num_train_epoches: int = field(default=3, metadata={'help': 'Total Epoch(1-3)'})
    per_device_train_batch_size: int = field(default=1, metadata={'help': 'Per Device Training Batch Size'})
    gradient_accumulation_steps: int = field(default=16, metadata={'help': 'Gradient Accumulation Steps'})

    eval_strategy: str = field(default="steps", metadata={"help": "Evaluation strategy: no, steps, epoch"})
    eval_steps: int = field(default=2, metadata={"help": "Run evaluation every n steps"})
    per_device_eval_batch_size: int = field(default=1, metadata={'help': 'Per Device Eval Batch Size'})

    optim: str = field(default='paged_adamw_32bit', metadata={'help': 'Optimization Method'})
    learning_rate: float = field(default=2e-4, metadata={'help': 'Learning Rate'})
    lr_scheduler_type: str = field(default='constant',
                                   metadata={'help': 'Learning Rate Scheduler: Constant, Linear, Cosine'})
    warmup_ratio: float = field(default=0.03, metadata={'help': 'Warmup Ratio'})

    lora_r: int = field(default=64, metadata={'help': 'LoRa R'})
    lora_alpha: float = field(default=16, metadata={'help': 'LoRa Alpha'})
    lora_dropout: float = field(default=0.0, metadata={'help': 'LoRa Dropout'})

    remove_unused_columns: bool = field(default=False, metadata={'help': 'Remove Unused Columns Or Not'})
    report_to: str = field(default='none', metadata={'help': 'Not Use Logger'})


@dataclass
class EvaluationArguments:
    checkpoint_dir: Optional[str] = field(default="llm_log", metadata={'help': 'Checkpoint Dir'})
    full_finetune: bool = field(default=False, metadata={'help': 'FineTurn Entire Model Without Adapter'})


@dataclass
class GenerationArguments:
    # 设置output长度
    max_new_tokens: Optional[int] = field(default=64, metadata={'help': 'Max New Tokens'})
    min_new_tokens: Optional[int] = field(default=1, metadata={'help': 'Min New Tokens'})

    # 设置生成策略(贪心搜索, 不惩罚相似内容)
    do_sample: Optional[bool] = field(default=True, metadata={'help': 'Sample Or Not'})
    num_beams: Optional[int] = field(default=1, metadata={'help': 'Num Beam'})
    num_beam_groups: Optional[int] = field(default=1, metadata={'help': 'Num Beam Groups'})
    penalty_alpha: Optional[float] = field(default=None, metadata={'help': 'Penalty Alpha'})
    use_cache: Optional[bool] = field(default=True, metadata={'help': 'Use Cache Or Not'})

    # 设置词概率处理(增强多样性, 禁止重复)
    temperature: Optional[float] = field(default=1.0, metadata={'help': 'Temperature'})
    top_k: Optional[int] = field(default=50, metadata={'help': 'Top K'})
    top_p: Optional[float] = field(default=0.9, metadata={'help': 'Top p'})
    typical_p: Optional[float] = field(default=1.0, metadata={'help': 'Typical p'})
    diversity_penalty: Optional[float] = field(default=0.0, metadata={'help': 'Diversity Penalty'})
    repetition_penalty: Optional[float] = field(default=1.0, metadata={'help': 'Repetition Penalty'})
    length_penalty: Optional[float] = field(default=1.0, metadata={'help': 'Length Penalty'})
    no_repeat_ngram_size: Optional[int] = field(default=0, metadata={})

    # 设置输出格式
    num_return_sequences: Optional[int] = field(default=1, metadata={'help': 'Num Return Sequences'})
    output_scores: Optional[bool] = field(default=False, metadata={'help': 'Return Output Scores Or Not'})
    return_dict_in_generate: Optional[bool] = field(default=True, metadata={
        'help': 'Return Dict(Sequences、 Scores、 Beam_Indices、 Sequence_Scores) In Generate Or Not'})


def get_logger(log_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_format = logging.Formatter('[%(asctime)s %(levelname)s] %(message)s')

    file_handler = logging.FileHandler(os.path.join(log_dir, 'logs.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(log_format)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def get_accelerate_model(args, config, pretrained_model_class):
    """
        启用模型量化 模型高效微调peft
    """
    print(f'Loading base model {args.model_name_or_path}...')
    # device_map = 'auto' if os.environ.get('LOCAL_RANK') is None else {'': int(os.environ.get('LOCAL_RANK', 0))}
    # 模型量化
    if args.use_quant:
        compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
        model = pretrained_model_class.from_pretrained(args.model_name_or_path, config=config,
                                                       device_map='auto',
                                                       quantization_config=BitsAndBytesConfig(
                                                           load_in_4bit=args.bits == 4,
                                                           load_in_8bit=args.bits == 8,
                                                           llm_int8_threshold=6.0,
                                                           llm_int8_has_fp16_weight=False,
                                                           bnb_4bit_compute_dtype=compute_dtype,
                                                           bnb_4bit_use_double_quant=args.double_quant,
                                                           bnb_4bit_quant_type=args.quant_type
                                                       ),
                                                       torch_dtype=compute_dtype)
    else:
        model = pretrained_model_class.from_pretrained(args.model_name_or_path, config=config, low_cpu_mem_usage=True,
                                                       device_map='auto')

    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    # 模型微调(高效微调Lora)
    if not args.full_finetune:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.use_quant)
        print(f'Adding LoRA modules...')
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias='none',
            task_type='CAUSAL_LM'
        )
        model = get_peft_model(model, config)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module.to(torch.bfloat16)
        if 'norm' in name:
            module.to(torch.float32)

        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module.to(torch.bfloat16)

    return model


class SavePeftModelCallback(transformers.TrainerCallback):

    def save_model(self, args, state, kwargs):
        print("Saving PEFT checkout...")
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint,
                                             'adapter_model')  # evaluation_strategy == 'no'
        else:
            checkpoint_folder = os.path.join(args.output_dir, f'checkpoint-{state.global_step}')

        peft_model_path = os.path.join(checkpoint_folder, 'adapter_model')
        kwargs['model'].save_pretrained(peft_model_path)

        # 节省磁盘空间，只保留你需要保存的 adapter 权重或特定内容（如 LoRA 的 kge 权重）
        for file_name in os.listdir(checkpoint_folder):
            if 'kge' in file_name:
                continue
            file_path = os.path.join(checkpoint_folder, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)

    def on_train_end(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
