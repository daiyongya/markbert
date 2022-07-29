# -*- coding: utf-8 -*-
# @Time    : 2021/6/7 14:25
# @Author  : Linyang Li
# @Email   : linyangli19@fudan.edu.cn
# @File    : markbert-tokenizer


from transformers import BertForSequenceClassification, BertModel, BertTokenizer, AutoConfig

class MarkBertTokenizer(BertTokenizer):

    def _tokenize(self, text):
        split_tokens = text.split(' ')
        return split_tokens

