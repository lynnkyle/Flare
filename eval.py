import argparse
import json
import os.path

import numpy as np
import torch
from peft import PeftModel

from dataset import KGDataset, KGDataModule
from tqdm import tqdm
from transformers import HfArgumentParser, GenerationConfig, AutoTokenizer, LlamaForCausalLM

from model import EmbeddingModel, KGELlama
from utils import ModelArguments, DataArguments, EvaluationArguments, GenerationArguments, get_logger


class Evaluator(object):
    def __init__(self, args, tokenizer, model, data_module, generation_config):
        self.sample_size = 200
        self.args = args
        self.tokenizer = tokenizer
        self.model = model
        self.data_module = data_module
        self.generation_config = generation_config

    @torch.no_grad()
    def eval_metric(self, dataset: KGDataset):
        self.model.eval()
        preds = []
        all_raw_ranks = []
        all_ranks = []

        generated = []

        # 生成预测
        for ex_idx, ex in enumerate(tqdm(dataset)):
            prompt = ex['input']
            if self.args.model_class == 'KGELlama':
                inputs = self.tokenizer(prompt, return_tensors='pt')
                input_ids = inputs['input_ids'].cuda()
                output = self.model.generate(
                    input_ids=input_ids,
                    query_ids=torch.LongTensor([ex['query_id']]).to(input_ids.device),
                    entity_ids=torch.LongTensor([ex['entity_ids']]).to(input_ids.device),
                    generation_config=self.generation_config
                )
                generated.append(output.sequences[0].cpu().numpy().tolist())
            else:
                raise NotImplementedError
            ex.pop('input')

        batch_preds = self.tokenizer.batch_decode(generated, skip_special_tokens=True)

        # 计算 rank
        for ex_idx, ex in enumerate(dataset):
            target = ex.pop('output')
            rank = ex['rank']
            pred = str(batch_preds[ex_idx]).strip()

            topK_names = ex['topk_names']
            if target == pred:
                rank = 1
            else:
                if pred in topK_names:
                    rank = min(rank, topK_names.index(pred) + 1)
                else:
                    rank = len(topK_names) + 1

            preds.append(ex)
            all_raw_ranks.append(ex['rank'])
            all_ranks.append(rank)

        # 全量计算指标
        def compute_metrics(rank_list):
            rank_arr = np.array(rank_list)
            metrics = {
                'hits1': np.mean(rank_arr <= 1),
                'hits3': np.mean(rank_arr <= 3),
                'hits10': np.mean(rank_arr <= 10),
                'mrr': np.mean(1. / rank_arr)
            }
            return {k: round(v, 3) for k, v in metrics.items()}

        raw_metrics = compute_metrics(all_raw_ranks)
        metrics = compute_metrics(all_ranks)

        # 打印最终指标
        logger.info('=' * 80)
        logger.info('Final raw_metrics: {}'.format(raw_metrics))
        logger.info('Final metrics: {}'.format(metrics))
        logger.info('=' * 80)

        return preds, raw_metrics, metrics


def print_parameter_datatypes(model, logger=None):
    dtypes = dict()
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()

    total = 0
    for k, v in dtypes.items(): total += v

    for k, v in dtypes.items():

        if logger is None:
            print(f'type: {k} || num: {v} || {round(v / total, 3)}')
        else:
            logger.info(f'type: {k} || num: {v} || {round(v / total, 3)}')


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    os.environ['NCCL_P2P_DISABLE'] = '1'
    os.environ['NCCL_IB_DISABLE'] = '1'
    torch.cuda.set_device(1)
    hfparser = HfArgumentParser((ModelArguments, DataArguments, EvaluationArguments, GenerationArguments))
    model_args, data_args, eval_args, generation_args, _ = hfparser.parse_args_into_dataclasses(
        return_remaining_strings=True)
    generation_config = GenerationConfig(**vars(generation_args))
    args = argparse.Namespace(**vars(model_args), **vars(data_args), **vars(eval_args))
    assert args.model_class in ['KGELlama']
    if args.kge_model == 'TransE':
        args.embedding_dim = 250

    logger = get_logger(os.path.dirname(args.checkpoint_dir))
    logger.info('args==>')
    logger.info(json.dumps(vars(args), ensure_ascii=False, indent=4))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    if args.model_class == 'KGELlama':
        tokenizer.add_tokens(['[QUERY]', '[ENTITY]'])
        model = LlamaForCausalLM.from_pretrained(args.model_name_or_path, low_cpu_mem_usage=True, device_map=None)
        model = PeftModel.from_pretrained(model, os.path.join(args.checkpoint_dir, "adapter_model")).cuda(1)
        llm_config = model.config
        kge_embedding_dir = os.path.join(args.dataset, args.kge_model)
        embed_model = EmbeddingModel(kge_embedding_dir, args.embedding_dim, 1024, llm_config.hidden_size,
                                     llm_config.hidden_act)
        embed_model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, 'kge_model.pth'), map_location='cpu'))
        embed_model.cuda()
        model = KGELlama(tokenizer, model, embed_model)

    model.eval()
    print_parameter_datatypes(model, logger)
    data_module = KGDataModule(args, tokenizer)
    evaluator = Evaluator(args, tokenizer, model, data_module, generation_config)
    preds, raw_metrics, metrics = evaluator.eval_metric(data_module.test_dataset)
    output = {
        'args': vars(args),
        'generation_config': generation_config.to_dict(),
        'predication': preds,
        'raw_metrics': raw_metrics,
        'metrics': metrics
    }
    output_path = os.path.join(os.path.dirname(args.checkpoint_dir), 'prediction.json')
    json.dump(output, open(output_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
