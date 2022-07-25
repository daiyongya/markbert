# -*- coding: utf-8 -*-
# @Time    : 2021/7/23 13:12
# @Author  : Linyang Li
# @Email   : linyangli19@fudan.edu.cn
# @File    : run_ner_flag
from fastNLP.modules import ConditionalRandomField
from transformers import BertModel, BertTokenizer, BertConfig

import sys
from load_data import load_ontonotes_cn
import torch
from fastNLP import Vocabulary, Trainer
from fastNLP import LossInForward
from fastNLP import BucketSampler
from fastNLP import FitlogCallback, EarlyStopCallback
import fitlog
from fastNLP import AccuracyMetric, SpanFPreRecMetric
from utils import Unfreeze_Callback
import os
if not os.path.exists('./nerlogs'):
    os.makedirs('./nerlogs')
if not os.path.exists('../ontonotes_mark/'):
    os.makedirs('../ontonotes_mark/')
if not os.path.exists('../msra_mark/'):
    os.makedirs('../msra_mark/')

fitlog.set_log_dir('./nerlogs')
ontonote4ner_cn_path = '../ontonotes_mark/'
msra_ner_cn_path = '../msra-mark/'
import argparse
import torch.optim as optim

from fastNLP import WarmupCallback
from transformers import AdamW
import math


class MyWarmupCallback(WarmupCallback):

    def __init__(self, warmup=0.1, schedule='constant'):
        """

        :param int,float warmup: 如果warmup为int，则在该step之前，learning rate根据schedule的策略变化; 如果warmup为float，
            如0.1, 则前10%的step是按照schedule策略调整learning rate。
        :param str schedule: 以哪种方式调整。
            linear: 前warmup的step上升到指定的learning rate(从Trainer中的optimizer处获取的), 后warmup的step下降到0；
            constant前warmup的step上升到指定learning rate，后面的step保持learning rate.
        """
        super().__init__()
        self.warmup = max(warmup, 0.)

        self.initial_lrs = []  # 存放param_group的learning rate
        if schedule == 'constant':
            self.get_lr = self._get_constant_lr
        elif schedule == 'linear':
            self.get_lr = self._get_linear_lr
        elif schedule == 'inverse_square':
            self.get_lr = self._get_inverse_square_lr
        else:
            raise RuntimeError("Only support 'linear', 'constant'.")

    def _get_inverse_square_lr(self, progress):
        if progress < self.warmup:
            return progress / self.warmup
        return max((math.sqrt(progress) - 1.) / (math.sqrt(self.warmup) - 1.), 0.)


parser = argparse.ArgumentParser()

parser.add_argument('--device', default=0, )
parser.add_argument('--demo', type=int, default=0)
parser.add_argument('--demo_train', type=int, default=0)
parser.add_argument('--debug', type=int, default=0)
parser.add_argument('--dataset', choices=['ontonotes_cn', 'msra_ner',
                                          'clue_ner'])
parser.add_argument('--encoding_type', choices=['bioes', 'bio', 'bmeso'], default='bmeso')
parser.add_argument('--seed', type=int, default=100)
parser.add_argument('--msg')
parser.add_argument('--if_save', type=int, default=0)

# parser.add_argument('--ptm_dropout',type=float,default=0.5)
parser.add_argument('--use_crf', default=0, type=int)
parser.add_argument('--ptm_name', default='hfl/chinese-bert-wwm')  # 适用于transformers，而不是FastNLP
parser.add_argument('--ptm_path', default='hfl/chinese-bert-wwm')  # 适用于transformers，而不是FastNLP

parser.add_argument('--after_bert', choices=['linear', 'tener'], default='linear')
# parser.add_argument('--ptm_layers',default='-1')
parser.add_argument('--ptm_pool_method', choices=['first', 'last', 'first_skip_space'], default='first')
parser.add_argument('--cls_hidden', type=int, default=128)
parser.add_argument('--cls_head', type=int, default=2)
parser.add_argument('--cls_ff', type=int, default=2)
parser.add_argument('--cls_dropout', type=float, default=0.3)
parser.add_argument('--cls_after_norm', type=int, default=1)
parser.add_argument('--cls_scale', type=int, default=0)
parser.add_argument('--cls_drop_attn', type=float, default=0.05)
parser.add_argument('--use_bigram', type=int, default=0)
parser.add_argument('--use_char', type=int, default=0)
parser.add_argument('--use_word', type=int, default=0)  # can try
parser.add_argument('--embed_dropout', type=float, default=0.3)
parser.add_argument('--keep_norm_same', type=int, default=0)
parser.add_argument('--word_embed_dim', type=int, default=100)
# parser.add_argument('--ptm_word_dropout',default=0.01,type=float)

parser.add_argument('--sampler', default='bucket', choices=['bucket', 'sequential', 'random'])
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd'])
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--ptm_lr_rate', type=float, default=1)
parser.add_argument('--crf_lr_rate', type=float, default=1)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=1e-2)
parser.add_argument('--warmup_step', type=float, default=3000)
parser.add_argument('--warmup_schedule', choices=['linear', 'constant', 'inverse_square'], default='inverse_square')
parser.add_argument('--early_stop_patience', type=int, default=-1)
parser.add_argument('--fix_ptm_epoch', type=int, default=-1)
parser.add_argument('--fix_ptm_step', type=int, default=-1)
parser.add_argument('--gradient_clip_norm_bert', type=float, default=1)
parser.add_argument('--gradient_clip_norm_other', type=float, default=5)  # english 15 , chinese 5

args = parser.parse_args()

if args.fix_ptm_epoch > 0 and args.fix_ptm_step > 0:
    print('only fix one, epoch or step! exit!')
    exit(1)

en_dataset = ['conll', 'ontonotes', 'twitter_ner', 'ark_twitter_pos', 'ritter_pos']
cn_dataset = ['ontonotes_cn', 'weibo', 'e_com', 'ctb5_pos', 'ctb7_pos', 'ctb9_pos', 'msra_ner', 'ud1_pos', 'ud2_pos', 'ud_seg', 'ctb5_seg', 'clue_ner']
en_ptm = ['albert-base-v2', 'bert-base-cased']
cn_ptm = ['hfl/chinese-bert-wwm']
language_ = 'none'

# if args.dataset in ['ritter_pos', 'ark_twitter_pos', 'weibo', 'clue_ner']:
#     args.epoch = 100
# else:
#     args.epoch = 40

args.language_ = language_
fitlog.set_rng_seed(args.seed)
args_to_save = args.__dict__.copy()
args_to_save['ptm_path'] = args.ptm_path.split('/')[-1]
fitlog.add_hyper(args_to_save)

if args.device not in ['cpu', 'all']:
    assert isinstance(args.device, int) or args.device.isdigit()
    device = torch.device('cuda:{}'.format(args.device))
elif args.device == 'cpu':
    device = torch.device('cpu')
elif args.device == 'all':
    device = [i for i in range(torch.cuda.device_count())]
else:
    raise NotImplementedError

if args.demo or args.demo_train:
    fitlog.debug()

if args.demo and args.demo_train:
    print(args.demo)
    print(args.demo_train)
    print('demo 和 demo_train 不能同时开，所以退出 exit')
    exit()

if args.device != 'cpu':
    assert isinstance(args.device, int) or args.device.isdigit()
    device = torch.device('cuda:{}'.format(args.device))
else:
    device = torch.device('cpu')

from model import BertNER
from transformers import AutoModel, AutoConfig, DistilBertModel, DistilBertConfig, BertModel

bert_config = BertConfig.from_pretrained(args.ptm_name)
bert_model = BertModel.from_pretrained(args.ptm_path, config=bert_config)

print(bert_config)
model_type = bert_config.model_type
print(bert_model)

refresh_data = False

cache_name = 'cache/{}_{}_{}'.format(args.dataset, args.encoding_type, model_type)
print('cache_name', cache_name)

if args.dataset == 'ontonotes':
    raise NotImplementedError
    bundle = load_ontonotes(ontonotes_path, args.encoding_type, _cache_fp=cache_name)
elif args.dataset == 'ontonotes_cn':
    bundle = load_ontonotes_cn(ontonote4ner_cn_path, args.encoding_type, pretrained_model_name_or_path=args.ptm_name,
                               dataset_name=args.dataset)
elif args.dataset == 'msra_ner':
    bundle = load_ontonotes_cn(msra_ner_cn_path, args.encoding_type, pretrained_model_name_or_path=args.ptm_name,
                               dataset_name=args.dataset)
else:
    print('不支持该数据集')
    exit()

for k, v in bundle.datasets.items():
    v.set_pad_val('words', bundle.vocabs['words'].padding_idx)

args.num_labels = len(bundle.get_vocab('target'))
model = BertNER(bert_model, bert_config, args)
print('*' * 20, 'param not ptm', '*' * 20)
for k, v in model.named_parameters():
    if 'ptm_encoder.' not in k:
        print('{}:{}'.format(k, v.size()))
print('*' * 20, 'param not ptm', '*' * 20)

params = {}
params['crf'] = []
params['ptm'] = []
params['ptm_no_decay'] = []
params['other'] = []
params['basic_embedding'] = []

params_name = {}
params_name['crf'] = []
params_name['ptm'] = []
params_name['ptm_no_decay'] = []
params_name['other'] = []
params_name['basic_embedding'] = []

for k, v in model.named_parameters():
    # print(k,v.size())
    if k[:len('ptm_encoder.')] == 'ptm_encoder.':
        if 'layer_norm' in k.lower() or 'layernorm' in k.lower() or 'bias' in k.lower():
            params['ptm_no_decay'].append(v)
            params_name['ptm_no_decay'].append(k)
        else:
            params['ptm'].append(v)
            params_name['ptm'].append(k)
    elif 'cnn_char.' in k or 'bigram_embedding.' in k or 'word_embedding.' in k:
        params['basic_embedding'].append(v)
        params_name['basic_embedding'].append(k)

    elif k[:len('crf.')] == 'crf.':
        params['crf'].append(v)
        params_name['crf'].append(k)
    else:
        params['other'].append(v)
        params_name['other'].append(k)

# exit()
param_ = [{'params': params['ptm'], 'lr': args.lr * args.ptm_lr_rate},
          {'params': params['ptm_no_decay'], 'lr': args.lr * args.ptm_lr_rate, 'weight_decay': 0},
          {'params': params['crf'], 'lr': args.lr * args.crf_lr_rate},
          {'params': params['other'], 'lr': args.lr},
          {'params': params['basic_embedding'], 'lr': args.lr}]

if args.optimizer == 'adam':
    # optimizer = optim.AdamW(param_,lr=args.lr,weight_decay=args.weight_decay)
    optimizer = AdamW(param_, lr=args.lr, weight_decay=args.weight_decay)
elif args.optimizer == 'sgd':
    optimizer = optim.SGD(param_, lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)

callbacks = []
# callbacks.append(FitlogCallback)
from fastNLP import GradientClipCallback

gradient_callback_bert = GradientClipCallback(params['ptm'], clip_value=args.gradient_clip_norm_bert, clip_type='norm')
gradient_callback_other = GradientClipCallback(params['other'] + params['crf'], clip_value=args.gradient_clip_norm_other, clip_type='norm')

callbacks.append(gradient_callback_bert)
callbacks.append(gradient_callback_other)

if args.warmup_step:
    callbacks.append(MyWarmupCallback(warmup=args.warmup_step, schedule=args.warmup_schedule))
if args.fix_ptm_epoch > 0:
    from utils import Unfreeze_Callback

    # model.ptm_encoder.requires_grad = False
    for k, v in model.ptm_encoder.named_parameters():
        v.requires_grad = False
    print('model param freeze!')
    # exit()
    callbacks.append(Unfreeze_Callback(model.ptm_encoder, fix_epoch_num=args.fix_ptm_epoch))
elif args.fix_ptm_step > 0:
    from utils import Unfreeze_Callback

    # model.ptm_encoder.requires_grad = False
    for k, v in model.ptm_encoder.named_parameters():
        v.requires_grad = False
    print('model param freeze!')
    # exit()
    callbacks.append(Unfreeze_Callback(model.ptm_encoder, fix_step_num=args.fix_ptm_step))
else:
    if hasattr(model.ptm_encoder, 'requires_grad'):
        assert model.ptm_encoder.requires_grad
if args.early_stop_patience > 0:
    callbacks.append(EarlyStopCallback(args.early_stop_patience))

metrics = []
# acc_metric = AccuracyMetric(pred='pred',target='target')
# acc_metric.set_metric_name('acc')
english_pos_dataset = ['ritter_pos', 'ark_twitter_pos']
if args.dataset not in english_pos_dataset:
    f_metric = SpanFPreRecMetric(bundle.vocabs['target'], 'pred', 'target', encoding_type=args.encoding_type)
    f_metric.set_metric_name('span')
    metrics.append(f_metric)
acc_metric = AccuracyMetric(pred='pred', target='target')
acc_metric.set_metric_name('acc_span')
metrics.append(acc_metric)

for k, v in bundle.datasets.items():
    v.set_input('words', 'seq_len', 'target', 'word_pieces', 'bert_attention_mask')
    if language_ == 'cn':
        if args.use_bigram:
            v.set_input('bigrams')
    if args.ptm_pool_method == 'first':
        v.set_input('first_word_pieces_pos')
    elif args.ptm_pool_method == 'first_skip_space':
        v.set_input('first_word_pieces_pos_skip_space')
    elif args.ptm_pool_method == 'last':
        v.set_input('last_word_pieces_pos')

    v.set_pad_val('bert_attention_mask', pad_val=0)
    if args.use_crf:
        v.set_pad_val('target', pad_val=0)
    else:
        v.set_pad_val('target', pad_val=-100)
    v.set_pad_val('first_word_pieces_pos', pad_val=0)
    v.set_pad_val('first_word_pieces_pos_skip_space', pad_val=0)
    v.set_pad_val('last_word_pieces_pos', pad_val=0)

    v.set_target('target', 'seq_len')

if bundle.datasets.get('dev') is not None:
    if bundle.datasets.get('test') is not None:
        fitlog_callback = FitlogCallback(data={'test': bundle.datasets['test'],
                                               'train': bundle.datasets['train'][:2000]}, log_loss_every=100)
else:
    fitlog_callback = FitlogCallback(data={'train': bundle.datasets['train'][:2000]}, log_loss_every=100)

callbacks.append(fitlog_callback)

from fastNLP import Callback


class Save_Model_Callback(Callback):
    def __init__(self, model, output_path):
        super().__init__()
        self.model_ = model
        self.output_path = output_path

    def on_valid_end(self, eval_result, metric_key, optimizer, is_better_eval):
        if True:
            f_model = open('{}/model/epoch_{}.pkl'.format(self.output_path, self.epoch), 'wb')
            torch.save(self.model_, f_model)
            f_model.close()


if args.if_save:
    from utils import get_peking_time

    exp_tag = get_peking_time()
    fitlog.add_other(str(exp_tag)[5:], 'time_tag')
    exp_path = 'exps/{}'.format(exp_tag)
    import os

    os.makedirs(exp_path, exist_ok=False)
    os.makedirs('{}/model'.format(exp_path), exist_ok=False)
    f_bundle = open('{}/bundle.pkl'.format(exp_path), 'wb')
    f_args = open('{}/args.pkl'.format(exp_path), 'wb')
    torch.save(bundle, f_bundle)
    torch.save(args, f_args)

    f_bundle.close()
    f_args.close()

    callbacks.append(Save_Model_Callback(model, exp_path))

if args.sampler == 'bucket':
    sampler = BucketSampler()
valid_steps = len(bundle.datasets['train']) // args.batch_size // 4
print('valid every {} steps'.format(valid_steps))
trainer = Trainer(bundle.datasets['train'], model, optimizer, LossInForward(), args.batch_size, sampler,
                  n_epochs=args.epoch,
                  dev_data=bundle.datasets.get('dev') if 'dev' in bundle.datasets else bundle.datasets['test'],
                  metrics=metrics, metric_key='f' if args.dataset not in english_pos_dataset else None,
                  callbacks=callbacks, device=device, num_workers=2, dev_batch_size=64, use_tqdm=False, print_every=100,
                  validate_every=valid_steps)
trainer.train()

fitlog.finish()
