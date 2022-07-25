from transformers import BertModel, BertConfig
import torch
import torch.nn as nn
from fastNLP import seq_len_to_mask

from utils import batch_index_select_yf, get_crf_zero_init

class BertNER(nn.Module):
    def __init__(self, bert_model, bert_config, args):
        super().__init__()
        self.ptm_encoder = bert_model
        self.config = bert_config
        self.args = args
        self.cls = nn.Linear(bert_config.hidden_size, args.num_labels)
        assert args.after_bert == 'linear'
        self.crf = None
        self.dropout = nn.Dropout(args.cls_dropout)
        if args.use_crf:
            self.crf = get_crf_zero_init(args.num_labels)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, word_pieces, bert_attention_mask, **inputs):
        input_ids = word_pieces
        attention_mask = bert_attention_mask
        seq_len = inputs['seq_len']

        ptm_pool_pos = None
        if self.args.ptm_pool_method == 'first':
            ptm_pool_pos = inputs['first_word_pieces_pos']
        elif self.args.ptm_pool_method == 'first_skip_space':
            ptm_pool_pos = inputs['first_word_pieces_pos_skip_space']
        elif self.args.ptm_pool_method == 'last':
            ptm_pool_pos = inputs['last_word_pieces_pos']

        target = inputs['target']

        bert_outputs = self.ptm_encoder(input_ids, attention_mask=attention_mask)
        rep = bert_outputs[0]

        rep = batch_index_select_yf(rep, ptm_pool_pos)
        rep = self.dropout(rep)

        logits = self.cls(rep)
        if self.crf is None:
            loss = self.loss_fn(logits.view(-1, self.args.num_labels), target.view(-1))
            pred = torch.argmax(logits, dim=-1)
            return {'loss': loss, 'pred': pred}
        else:
            mask = seq_len_to_mask(seq_len)
            loss = self.crf(logits, target, mask).mean(dim=0)
            pred, _ = self.crf.viterbi_decode(logits, mask)
            return {'loss': loss, 'pred': pred}
