# -*- coding: utf-8 -*-
# @Time    : 2021/6/7 14:25
# @Author  : Linyang Li
# @Email   : linyangli19@fudan.edu.cn
# @File    : markbert-tokenizer


from transformers import BertForSequenceClassification, BertModel, BertTokenizer, AutoConfig




class MarkBertTokenizer(BertTokenizer):

    def __init__(
            self,
            vocab_file,
            do_lower_case=True,
            do_basic_tokenize=True,
            never_split=None,
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
            tokenize_chinese_chars=True,
            strip_accents=None,
            **kwargs
    ):
        super().__init__(
            vocab_file,
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )

        self.encode_type = 1
        self.separate_token = '[unused1]'
        self.postags = {'vv': '[unused6]',
                        'cd': '[unused7]',
                        'm': '[unused8]',
                        'nn': '[unused9]',
                        'pu': '[unused10]',
                        'jj': '[unused11]',
                        'ad': '[unused12]',
                        'as': '[unused13]',
                        'deg': '[unused14]',
                        'cc': '[unused15]',
                        'cs': '[unused16]',
                        've': '[unused17]',
                        'p': '[unused18]',
                        'nr': '[unused19]',
                        'sb': '[unused20]',
                        'dec': '[unused21]',
                        'lc': '[unused22]',
                        'dt': '[unused23]',
                        'nt': '[unused24]',
                        'vc': '[unused25]',
                        'va': '[unused26]',
                        'ba': '[unused27]',
                        'od': '[unused28]',
                        'pn': '[unused29]',
                        'sp': '[unused30]',
                        'etc': '[unused31]',
                        }

    def _tokenize(self, text):
        split_tokens = text.split(' ')

        if self.encode_type == 1:
            # word-to-char
            # with sep
            final_tokens = []
            for token in split_tokens:
                for char in token:
                    final_tokens.append(char)
                final_tokens.append(self.separate_token)

            return final_tokens

        if self.encode_type == 2:
            # word-to-char
            # with sep
            final_tokens = []
            for i, token in enumerate(split_tokens):

                if i % 2 == 0:
                    for i, char in enumerate(token):
                        final_tokens.append(char)
                else:
                    if token.lower() in self.postags:
                        final_tokens.append(self.postags[token.lower()])
                    else:
                        final_tokens.append(self.separate_token)

            return final_tokens

        return split_tokens

