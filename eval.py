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
        raw_ranks = np.array([])
        ranks = np.array([])
        print_step = 1000
        data_num = len(dataset)

        for begin_idx in range(0, data_num, print_step):
            end_idx = min(begin_idx + print_step, data_num)
            generated = []
            for ex_idx, ex in enumerate(tqdm(dataset[begin_idx:end_idx])):
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
            for ex_idx, ex in enumerate(dataset[begin_idx:end_idx]):
                target = ex.pop('output')
                rank = ex['rank']
                pred = str(batch_preds[ex_idx]).strip()

                topK_names = ex['topk_names']
                if target == pred:
                    rank = 1
                else:
                    if pred not in set(topK_names) or topK_names.index(pred) >= rank:
                        rank += 1

                # ex['target'] = target
                # ex['pred_rank'] = rank
                # ex['pred'] = pred
                preds.append(ex)
                raw_ranks = np.append(raw_ranks, ex['rank'])
                ranks = np.append(ranks, rank)

            def compute_metrics(rank):
                metrics = {
                    'hits1': np.mean(rank <= 1),
                    'hits3': np.mean(rank <= 3),
                    'hits10': np.mean(rank <= 10),
                    'mrr': np.mean(1. / rank)
                }
                metrics = {k: round(v, 3) for k, v in metrics.items()}
                return metrics

            logger.info('=' * 80)
            raw_metrics = compute_metrics(raw_ranks)
            logger.info('raw_metrics: {}'.format(raw_metrics))
            metrics = compute_metrics(ranks)
            logger.info('metrics: {}'.format(metrics))
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
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['NCCL_P2P_DISABLE'] = '1'
    os.environ['NCCL_IB_DISABLE'] = '1'
    torch.cuda.set_device(0)
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
        model = LlamaForCausalLM.from_pretrained(args.model_name_or_path, low_cpu_mem_usage=True, device_map='auto')
        model = PeftModel.from_pretrained(model, os.path.join(args.checkpoint_dir, "adapter_model"))
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
