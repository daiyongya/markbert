from fastNLP.io.pipe import Conll2003NERPipe, OntoNotesNERPipe
from fastNLP import cache_results
from transformers import AutoTokenizer, AlbertTokenizer, BertTokenizer, DistilBertTokenizer
from fastNLP import Vocabulary
import tqdm
import copy


# @cache_results(_cache_fp='cache/ontonotes4ner', _refresh=False)
def load_ontonotes4ner(path, char_embedding_path=None, bigram_embedding_path=None, index_token=True,
                       char_min_freq=1, bigram_min_freq=1, only_train_min_freq=0, norm_embed=False, encoding_type='bmeso'):
    assert encoding_type in ['bmeso', 'bioes', 'bio']
    from fastNLP.io.loader import ConllLoader
    from utils import get_bigrams
    import os
    from fastNLP import Vocabulary
    from fastNLP.embeddings import StaticEmbedding
    from fastNLP.io.data_bundle import DataBundle

    train_path = os.path.join(path, 'train.char.bmes{}'.format(''))
    dev_path = os.path.join(path, 'dev.char.bmes')
    test_path = os.path.join(path, 'test.char.bmes')

    loader = ConllLoader(['chars', 'target'])
    train_bundle = loader.load(train_path)
    dev_bundle = loader.load(dev_path)
    test_bundle = loader.load(test_path)

    datasets = dict()
    datasets['train'] = train_bundle.datasets['train']
    datasets['dev'] = dev_bundle.datasets['train']
    datasets['test'] = test_bundle.datasets['train']

    datasets['train'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')
    datasets['dev'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')
    datasets['test'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')

    datasets['train'].add_seq_len('chars')
    datasets['dev'].add_seq_len('chars')
    datasets['test'].add_seq_len('chars')

    char_vocab = Vocabulary()
    bigram_vocab = Vocabulary()
    label_vocab = Vocabulary(padding=None, unknown=None)
    # print(datasets.keys())
    # print(len(datasets['dev']))
    # print(len(datasets['test']))
    # print(len(datasets['train']))
    char_vocab.from_dataset(datasets['train'], field_name='chars',
                            no_create_entry_dataset=[datasets['dev'], datasets['test']])
    bigram_vocab.from_dataset(datasets['train'], field_name='bigrams',
                              no_create_entry_dataset=[datasets['dev'], datasets['test']])
    if 'ctb9' in path:
        label_vocab.from_dataset(datasets['train'], datasets['dev'], datasets['test'], field_name='target')
    else:
        label_vocab.from_dataset(datasets['train'], field_name='target')
    if index_token:
        for k, v in datasets.items():
            v.rename_field('chars', 'raw_chars')
            v.rename_field('bigrams', 'raw_bigrams')
        char_vocab.index_dataset(datasets['train'], datasets['dev'], datasets['test'],
                                 field_name='raw_chars', new_field_name='chars')
        bigram_vocab.index_dataset(datasets['train'], datasets['dev'], datasets['test'],
                                   field_name='raw_bigrams', new_field_name='bigrams')
        label_vocab.index_dataset(datasets['train'], datasets['dev'], datasets['test'],
                                  field_name='target', new_field_name='target')

    vocabs = {}
    vocabs['words'] = char_vocab
    # vocabs['label'] = label_vocab
    vocabs['bigrams'] = bigram_vocab
    vocabs['target'] = label_vocab

    embeddings = {}
    if char_embedding_path is not None:
        char_embedding = StaticEmbedding(char_vocab, char_embedding_path, word_dropout=0.01,
                                         min_freq=char_min_freq, only_train_min_freq=only_train_min_freq, normalize=norm_embed)
        embeddings['char'] = char_embedding

    if bigram_embedding_path is not None:
        bigram_embedding = StaticEmbedding(bigram_vocab, bigram_embedding_path, word_dropout=0.01,
                                           min_freq=bigram_min_freq, only_train_min_freq=only_train_min_freq, normalize=norm_embed)
        embeddings['bigram'] = bigram_embedding

    for k, v in datasets.items():
        v.rename_field('chars', 'words')
        v.rename_field('raw_chars', 'raw_words')
    bundle = DataBundle()
    bundle.datasets = datasets
    bundle.vocabs = vocabs
    bundle.embeddings = embeddings
    from utils import transform_bmeso_bundle_to_bio
    if encoding_type == 'bmeso':
        return bundle
    elif encoding_type == 'bio':
        bundle = transform_bmeso_bundle_to_bio(bundle)
        return bundle
    elif encoding_type == 'bioes':
        return bundle


from fastNLP.io.loader import ConllLoader
from fastNLP import DataSet, Instance
from fastNLP.core import logger


# @cache_results(_cache_fp='cache/msra_ner', _refresh=False)
def load_msra_ner(path, char_embedding_path=None, bigram_embedding_path=None, index_token=True,
                  char_min_freq=1, bigram_min_freq=1, only_train_min_freq=0, norm_embed=False, encoding_type='bmeso', train_clip=True):
    assert encoding_type in ['bmeso', 'bioes', 'bio']
    from fastNLP.io.loader import ConllLoader
    from utils import get_bigrams
    import os
    from fastNLP import Vocabulary
    from fastNLP.embeddings import StaticEmbedding
    from fastNLP.io.data_bundle import DataBundle

    if train_clip:
        train_path = os.path.join(path, 'train_dev_x.char.bmes_clip1')
        test_path = os.path.join(path, 'test_x.char.bmes_clip1')
    else:
        train_path = os.path.join(path, 'train_dev_x.char.bmes')
        test_path = os.path.join(path, 'test_x.char.bmes')

    loader = ConllLoader(['chars', 'target'])
    train_bundle = loader.load(train_path)
    # dev_bundle = loader.load(dev_path)
    test_bundle = loader.load(test_path)

    datasets = dict()
    datasets['train'] = train_bundle.datasets['train']
    # datasets['dev'] = dev_bundle.datasets['train']
    datasets['test'] = test_bundle.datasets['train']

    datasets['train'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')
    # datasets['dev'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')
    datasets['test'].apply_field(get_bigrams, field_name='chars', new_field_name='bigrams')

    datasets['train'].add_seq_len('chars')
    # datasets['dev'].add_seq_len('chars')
    datasets['test'].add_seq_len('chars')

    char_vocab = Vocabulary()
    bigram_vocab = Vocabulary()
    label_vocab = Vocabulary(padding=None, unknown=None)
    # print(datasets.keys())
    # print(len(datasets['dev']))
    # print(len(datasets['test']))
    # print(len(datasets['train']))
    char_vocab.from_dataset(datasets['train'], field_name='chars',
                            no_create_entry_dataset=[datasets['test']])
    bigram_vocab.from_dataset(datasets['train'], field_name='bigrams',
                              no_create_entry_dataset=[datasets['test']])
    if 'ctb9' in path:
        label_vocab.from_dataset(datasets['train'], datasets['test'], field_name='target')
    else:
        label_vocab.from_dataset(datasets['train'], field_name='target')
    if index_token:
        for k, v in datasets.items():
            v.rename_field('chars', 'raw_chars')
            v.rename_field('bigrams', 'raw_bigrams')
        char_vocab.index_dataset(datasets['train'], datasets['test'],
                                 field_name='raw_chars', new_field_name='chars')
        bigram_vocab.index_dataset(datasets['train'], datasets['test'],
                                   field_name='raw_bigrams', new_field_name='bigrams')
        label_vocab.index_dataset(datasets['train'], datasets['test'],
                                  field_name='target', new_field_name='target')

    vocabs = {}
    vocabs['words'] = char_vocab
    # vocabs['label'] = label_vocab
    vocabs['bigrams'] = bigram_vocab
    vocabs['target'] = label_vocab

    embeddings = {}
    if char_embedding_path is not None:
        char_embedding = StaticEmbedding(char_vocab, char_embedding_path, word_dropout=0.01,
                                         min_freq=char_min_freq, only_train_min_freq=only_train_min_freq, normalize=norm_embed)
        embeddings['char'] = char_embedding

    if bigram_embedding_path is not None:
        bigram_embedding = StaticEmbedding(bigram_vocab, bigram_embedding_path, word_dropout=0.01,
                                           min_freq=bigram_min_freq, only_train_min_freq=only_train_min_freq, normalize=norm_embed)
        embeddings['bigram'] = bigram_embedding

    for k, v in datasets.items():
        v.rename_field('chars', 'words')
        v.rename_field('raw_chars', 'raw_words')
    bundle = DataBundle()
    bundle.datasets = datasets
    bundle.vocabs = vocabs
    bundle.embeddings = embeddings
    # bundle.datasets['test']
    from utils import transform_bmeso_bundle_to_bio
    if encoding_type == 'bmeso':
        return bundle
    elif encoding_type == 'bio':
        bundle = transform_bmeso_bundle_to_bio(bundle)
        return bundle
    elif encoding_type == 'bioes':
        return bundle


# @cache_results(_cache_fp='tmp_ontonotes_cn',_refresh=False)
def load_ontonotes_cn(fp, encoding_type='bio', pretrained_model_name_or_path=None, dataset_name=''):
    assert dataset_name != ''
    # cache_name = 'cache/{}_{}_1'.format(dataset_name,encoding_type)
    cache_name = None
    if dataset_name == 'ontonotes_cn':
        bundle = load_ontonotes4ner(fp, index_token=True, char_min_freq=1, bigram_min_freq=1, norm_embed=True, encoding_type=encoding_type)
    elif 'msra_ner' == dataset_name:
        bundle = load_msra_ner(fp, index_token=True, char_min_freq=1, bigram_min_freq=1, encoding_type=encoding_type, norm_embed=True)


    else:
        raise NotImplementedError

    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path)

    # tokenizer.add_tokens(['`', '@'])
    # odd_words = set()
    word_to_wordpieces = []
    word_pieces_lengths = []
    vocab = bundle.vocabs['words']
    non_space_token = ['.', ',', ';', '%', ']', ')', '?', '!', '"', "'", '/', ':']
    for word, index in vocab:
        if index == vocab.padding_idx:  # pad是个特殊的符号
            word = tokenizer._pad_token
            print('pad:{}'.format(word))
        elif index == vocab.unknown_idx:
            word = tokenizer._unk_token
            print('unk:{}'.format(word))
        # elif vocab.word_count[word] < min_freq:
        #     word = '[UNK]'
        word_pieces = tokenizer.wordpiece_tokenizer.tokenize(word)
        # word_pieces = tokenizer.convert_ids_to_tokens(word_pieces)

        # if word_pieces[0] == '▁':
        #     word_pieces_ = word_pieces[1:]
        #     if len(word_pieces_) == 1 and len(word_pieces_[0]) == 1 and word_pieces_[0] in non_space_token:
        #         word_pieces = word_pieces_
        #     else:
        #         if word_pieces[0][0] != '▁':
        #             print('第一个token非空格，但开头不是空格：{}'.format(word_pieces))
        #         word_pieces = word_pieces
        #         # odd_words.add((word,tuple(word_pieces)))
        # else:
        #     word_pieces = word_pieces

        word_to_wordpieces.append(word_pieces)
        word_pieces_lengths.append(len(word_pieces))

    ins_num = 0
    for k, v in bundle.datasets.items():
        ins_num += len(v)
    pbar = tqdm.tqdm(total=ins_num)

    def get_word_pieces_bert_cn(ins):
        words = ins['words']
        raw_words = ins['raw_words']
        word_pieces = []
        raw_word_pieces = []
        now_ins_word_piece_length = []
        first_word_pieces_pos = []

        for i, w in enumerate(words):
            rwp = word_to_wordpieces[w]
            wp = tokenizer.convert_tokens_to_ids(rwp)
            # rwp = tokenizer.tokenize(rw)
            word_pieces.extend(wp)
            raw_word_pieces.extend(rwp)
            now_ins_word_piece_length.append(len(wp))

        for i, l in enumerate(now_ins_word_piece_length):
            if i == 0:
                first_word_pieces_pos.append(0)
            else:
                first_word_pieces_pos.append(first_word_pieces_pos[-1] + now_ins_word_piece_length[i - 1])

        assert len(first_word_pieces_pos) == len(words)

        first_word_pieces_pos_skip_space = copy.deepcopy(first_word_pieces_pos)
        # for i,pos in enumerate(first_word_pieces_pos):
        #     if raw_word_pieces[pos] == '▁':
        #         first_word_pieces_pos_skip_space.append(first_word_pieces_pos[i]+1)
        #     else:
        #         first_word_pieces_pos_skip_space.append(first_word_pieces_pos[i])

        last_word_pieces_pos = []
        for i, l in enumerate(now_ins_word_piece_length):
            if i == 0:
                last_word_pieces_pos.append(now_ins_word_piece_length[0] - 1)
            else:
                last_word_pieces_pos.append(last_word_pieces_pos[-1] + now_ins_word_piece_length[i])

        # add cls sep
        raw_word_pieces.append(tokenizer.sep_token)
        raw_word_pieces.insert(0, tokenizer.cls_token)
        word_pieces.append(tokenizer.sep_token_id)
        word_pieces.insert(0, tokenizer.cls_token_id)
        first_word_pieces_pos = list(map(lambda x: x + 1, first_word_pieces_pos))
        first_word_pieces_pos_skip_space = list(map(lambda x: x + 1, first_word_pieces_pos_skip_space))
        last_word_pieces_pos = list(map(lambda x: x + 1, last_word_pieces_pos))
        pbar.update(1)
        return raw_word_pieces, word_pieces, now_ins_word_piece_length, first_word_pieces_pos, first_word_pieces_pos_skip_space, last_word_pieces_pos

    for k, v in bundle.datasets.items():
        v.apply(get_word_pieces_bert_cn, 'tmp')
        v.apply_field(lambda x: x[0], 'tmp', 'raw_word_pieces')
        v.apply_field(lambda x: x[1], 'tmp', 'word_pieces')
        v.apply_field(lambda x: x[2], 'tmp', 'word_piece_num')  # 每个位置的词被拆解为了多少个word piece
        v.apply_field(lambda x: x[3], 'tmp', 'first_word_pieces_pos')  # 每个位置的词的第一个word piece在 word piece 序列中的位置
        v.apply_field(lambda x: x[4], 'tmp', 'first_word_pieces_pos_skip_space')  # 每个位置的词的第一个word piece在 word piece 序列中的位置，如果有空格，就加一
        v.apply_field(lambda x: x[5], 'tmp', 'last_word_pieces_pos')  # 每个位置的词的最后一个word piece在 word piece 序列中的位置，如果有空格，就加一
        v.apply_field(len, 'word_pieces', 'word_piece_seq_len')
        v.apply_field(lambda x: [1] * x, 'word_piece_seq_len', 'bert_attention_mask')

    bundle.tokenizer = tokenizer

    return bundle
