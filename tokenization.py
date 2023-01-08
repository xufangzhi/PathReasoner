# encoding=utf-8
# partly modified from DAGN

from dataclasses import dataclass, field
import argparse
from transformers import AutoTokenizer
import gensim
import re
import numpy as np
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from itertools import groupby
from operator import itemgetter
import copy
stemmer = SnowballStemmer("english")
punctuations = [',', '.', ';', ':']
stopwords = list(gensim.parsing.preprocessing.STOPWORDS) + punctuations


def has_same_logical_component(set1, set2):
    has_same = False
    overlap = -1
    if len(set1) > 1 and len(set2) > 1:
        overlap = len(set1 & set2)/max(min(len(set1), len(set2)), 1)
        if overlap > 0.5:  # hyper-parameter:0.5
            has_same = 1
    return has_same, overlap


def token_stem(token):
    return stemmer.stem(token)

def get_node_tag(bpe_tokens):
    i = 0
    mask_tag, tag_now = 0, 0
    cond_tag, res_tag = 1, 2
    node_tag = []
    while i < len(bpe_tokens):
        if bpe_tokens[i] == "<cond>" or bpe_tokens[i] == "<mask>" or bpe_tokens[i] == "<unk>":
            tag_now += 1
            # node_tag.append(tag_now)
            if bpe_tokens[i] == "<mask>":
                node_tag.append(cond_tag)
            else:
                node_tag.append(res_tag)
            bpe_tokens.pop(i)
            i += 1
        elif bpe_tokens[i] == "</cond>" or bpe_tokens[i] == "</s>":
            bpe_tokens.pop(i)
        else:
            node_tag.append(mask_tag)
            i += 1
    return bpe_tokens, node_tag



def arg_tokenizer(text_a, text_b, text_c, tokenizer, stopwords, relations:dict, punctuations:list,
                  max_gram:int, max_length:int, do_lower_case:bool=False):
    '''
    :param text_a: str. (context in a sample.)
    :param text_b: str. ([#1] option in a sample. [#2] question + option in a sample.)
    :param text_c: str. (question in a sample)
    :param tokenizer: RoBERTa tokenizer.
    :param relations: dict. {argument words: pattern}
    :return:
        - input_ids: list[int]
        - attention_mask: list[int]
        - segment_ids: list[int]
        - argument_bpe_ids: list[int]. value={ -1: padding,
                                                0: non_arg_non_dom,
                                                1: (relation, head, tail)  关键词在句首
                                                2: (head, relation, tail)  关键词在句中，先因后果
                                                3: (tail, relation, head)  关键词在句中，先果后因
                                                }
        - domain_bpe_ids: list[int]. value={ -1: padding,
                                              0:non_arg_non_dom,
                                           D_id: domain word ids.}
        - punctuation_bpe_ids: list[int]. value={ -1: padding,
                                                   0: non_punctuation,
                                                   1: punctuation}
    '''

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    def _is_stopwords(word, stopwords):
        if word in stopwords:
            is_stopwords_flag = True
        else:
            is_stopwords_flag = False
        return is_stopwords_flag

    def _head_tail_is_stopwords(span, stopwords):
        if span[0] in stopwords or span[-1] in stopwords:
            return True
        else:
            return False

    def _with_septoken(ngram, tokenizer):
        if tokenizer.bos_token in ngram or tokenizer.sep_token in ngram or tokenizer.eos_token in ngram:
            flag = True
        else: flag = False
        return flag

    def _is_argument_words(seq, argument_words):
        pattern = None
        arg_words = list(argument_words.keys())
        if seq.strip() in arg_words:
            pattern = argument_words[seq.strip()]
        return pattern

    def _is_exist(exists:list, start:int, end:int):
        flag = False
        for estart, eend in exists:
            if estart <= start and eend >= end:
                flag = True
                break
        return flag

    def _find_punct(tokens, punctuations):
        punct_ids = [0] * len(tokens)
        for i, token in enumerate(tokens):
            if token in punctuations:
                punct_ids[i] = 1
        return punct_ids

    def _find_arg_ngrams(tokens, max_gram):
        n_tokens = len(tokens)
        global_arg_start_end = []
        argument_words = {}
        argument_ids = [0] * n_tokens
        for n in range(max_gram, 0, -1):  # loop over n-gram.
            for i in range(n_tokens - n):  # n-gram window sliding.
                window_start, window_end = i, i + n
                ngram = " ".join(tokens[window_start:window_end])
                pattern = _is_argument_words(ngram, relations)
                if pattern:
                    if not _is_exist(global_arg_start_end, window_start, window_end):
                        global_arg_start_end.append((window_start, window_end))
                        argument_ids[window_start:window_end] = [pattern] * (window_end - window_start)
                        argument_words[ngram] = (window_start, window_end)

        return argument_words, argument_ids

    def _find_dom_ngrams_2(tokens, max_gram):
        '''
        1. 判断 stopwords 和 sep token
        2. 先遍历一遍，记录 n-gram 的重复次数和出现位置
        3. 遍历记录的 n-gram, 过滤掉 n-gram 子序列（直接比较 str）
        4. 赋值 domain_ids.

        '''

        stemmed_tokens = [token_stem(token) for token in tokens]

        ''' 1 & 2'''
        n_tokens = len(tokens)
        d_ngram = {}
        domain_words_stemmed = {}
        domain_words_orin = {}
        domain_ids = [0] * n_tokens
        for n in range(max_gram, 0, -1):  # loop over n-gram.
            for i in range(n_tokens - n):  # n-gram window sliding.

                window_start, window_end = i, i+n
                stemmed_span = stemmed_tokens[window_start:window_end]
                stemmed_ngram = " ".join(stemmed_span)
                orin_span = tokens[window_start:window_end]
                orin_ngram = " ".join(orin_span)

                if _is_stopwords(orin_ngram, stopwords): continue
                if _head_tail_is_stopwords(orin_span, stopwords): continue
                if _with_septoken(orin_ngram, tokenizer): continue

                if not stemmed_ngram in d_ngram:
                    d_ngram[stemmed_ngram] = []
                d_ngram[stemmed_ngram].append((window_start, window_end))

        ''' 3 '''
        d_ngram = dict(filter(lambda e: len(e[1]) > 1, d_ngram.items()))
        raw_domain_words = list(d_ngram.keys())
        raw_domain_words.sort(key=lambda s: len(s), reverse=True)  # sort by len(str).
        domain_words_to_remove = []
        for i in range(0, len(d_ngram)):
            for j in range(i+1, len(d_ngram)):
                if raw_domain_words[i] in raw_domain_words[j]:
                    domain_words_to_remove.append(raw_domain_words[i])
                if raw_domain_words[j] in raw_domain_words[i]:
                    domain_words_to_remove.append(raw_domain_words[j])
        for r in domain_words_to_remove:
            try:
                del d_ngram[r]
            except:
                pass

        ''' 4 '''
        d_id = 0
        for stemmed_ngram, start_end_list in d_ngram.items():
            d_id += 1
            for start, end in start_end_list:
                domain_ids[start:end] = [d_id] * (end - start)
                rebuilt_orin_ngram = " ".join(tokens[start: end])
                if not stemmed_ngram in domain_words_stemmed:
                    domain_words_stemmed[stemmed_ngram] = []
                if not rebuilt_orin_ngram in domain_words_orin:
                    domain_words_orin[rebuilt_orin_ngram] = []
                domain_words_stemmed[stemmed_ngram] +=  [(start, end)]
                domain_words_orin[rebuilt_orin_ngram] += [(start, end)]


        return domain_words_stemmed, domain_words_orin, domain_ids


    ''' start '''
    bpe_tokens_a = tokenizer.tokenize(text_a)
    bpe_tokens_b = tokenizer.tokenize(text_b)
    bpe_tokens_c = tokenizer.tokenize(text_c)

    bpe_tokens = [tokenizer.bos_token] + bpe_tokens_a + [tokenizer.sep_token] + \
                    bpe_tokens_b + [tokenizer.sep_token] + \
                    bpe_tokens_c + [tokenizer.eos_token]

    a_mask = [1] * (len(bpe_tokens_a) + 2) + [0] * (max_length - (len(bpe_tokens_a) + 2))
    b_mask = [0] * (len(bpe_tokens_a) + 2) + [1] * (len(bpe_tokens_b) + 1) + [0] * (max_length - (len(bpe_tokens_a) + 2 + len(bpe_tokens_b) + 1))
    c_mask = [0] * (len(bpe_tokens_a) + 2) + [0] * (len(bpe_tokens_b) + 1) + [1] * (len(bpe_tokens_c) + 1) + [0] * (max_length - len(bpe_tokens))

    a_mask = a_mask[:max_length]
    b_mask = b_mask[:max_length]
    c_mask = c_mask[:max_length]
    assert len(a_mask) == max_length, 'len_a_mask={}, max_len={}'.format(len(a_mask), max_length)
    assert len(b_mask) == max_length, 'len_b_mask={}, max_len={}'.format(len(b_mask), max_length)
    assert len(c_mask) == max_length, 'len_c_mask={}, max_len={}'.format(len(c_mask), max_length)

    if len(bpe_tokens)>max_length:
        bpe_tokens = bpe_tokens[:max_length]
    # adapting Ġ.
    assert isinstance(bpe_tokens, list)
    bare_tokens = [token[1:] if "Ġ" in token else token for token in bpe_tokens]
   
    argument_words, argument_space_ids = _find_arg_ngrams(bare_tokens, max_gram=max_gram)
    domain_words_stemmed, domain_words_orin, domain_space_ids = _find_dom_ngrams_2(bare_tokens, max_gram=max_gram)
    punct_space_ids = _find_punct(bare_tokens, punctuations)

    argument_bpe_ids = argument_space_ids
    domain_bpe_ids = domain_space_ids
    punct_bpe_ids = punct_space_ids

    ''' output items '''
    input_ids = tokenizer.convert_tokens_to_ids(bpe_tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * (len(bpe_tokens_a) + 2) + [1] * (len(bpe_tokens_b) + 1)
    padding = [0] * (max_length - len(input_ids))
    padding_ids = [tokenizer.pad_token_id] * (max_length - len(input_ids))
    arg_dom_padding_ids = [-1] * (max_length - len(input_ids))
    input_ids += padding_ids
    argument_bpe_ids += arg_dom_padding_ids
    domain_bpe_ids += arg_dom_padding_ids
    punct_bpe_ids += arg_dom_padding_ids
    input_mask += padding
    segment_ids += padding

    if len(input_ids) > max_length:
        print(len(input_ids))
    input_ids = input_ids[:max_length]
    input_mask = input_mask[:max_length]
    segment_ids = segment_ids[:max_length]
    argument_bpe_ids = argument_bpe_ids[:max_length]
    domain_bpe_ids = domain_bpe_ids[:max_length]
    punct_bpe_ids = punct_bpe_ids[:max_length]

    assert len(input_ids) <= max_length, 'len_input_ids={}, max_length={}'.format(len(input_ids), max_length)
    assert len(input_mask) <= max_length, 'len_input_mask={}, max_length={}'.format(len(input_mask), max_length)
    assert len(segment_ids) <= max_length, 'len_segment_ids={}, max_length={}'.format(len(segment_ids), max_length)
    assert len(argument_bpe_ids) <= max_length, 'len_argument_bpe_ids={}, max_length={}'.format(
        len(argument_bpe_ids), max_length)
    assert len(domain_bpe_ids) <= max_length, 'len_domain_bpe_ids={}, max_length={}'.format(
        len(domain_bpe_ids), max_length)
    assert len(punct_bpe_ids) <= max_length, 'len_punct_bpe_ids={}, max_length={}'.format(
        len(punct_bpe_ids), max_length)

    # print(punct_bpe_ids)

    output = {}
    output["input_tokens"] = bpe_tokens
    output["input_ids"] = input_ids
    output["attention_mask"] = input_mask
    output["token_type_ids"] = segment_ids
    output["argument_bpe_ids"] = argument_bpe_ids
    output["domain_bpe_ids"] = domain_bpe_ids
    output["punct_bpe_ids"] = punct_bpe_ids
    output["a_mask"] = a_mask
    output["b_mask"] = b_mask
    output["c_mask"] = c_mask

    return output


def logic_tokenizer(text_a, text_b, text_c, tokenizer, stopwords, relations:dict, punctuations:list,
                  max_gram:int, max_length:int, do_lower_case:bool=False):
    '''
    :param text_a: str. (context in a sample.)
    :param text_b: str. ([#1] option in a sample. [#2] question + option in a sample.)
    :param text_c: str. (question in a sample)
    :param tokenizer: RoBERTa tokenizer.
    :param relations: dict. {argument words: pattern}
    :return:
        - input_ids: list[int]
        - attention_mask: list[int]
        - segment_ids: list[int]
        - argument_bpe_ids: list[int]. value={ -1: padding,
                                                0: non_arg_non_dom,
                                                1: (relation, head, tail)  关键词在句首
                                                2: (head, relation, tail)  关键词在句中，先因后果
                                                3: (tail, relation, head)  关键词在句中，先果后因
                                                }
        - domain_bpe_ids: list[int]. value={ -1: padding,
                                              0:non_arg_non_dom,
                                           D_id: domain word ids.}
        - punctuation_bpe_ids: list[int]. value={ -1: padding,
                                                   0: non_punctuation,
                                                   1: punctuation}
    '''

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    def _is_stopwords(word, stopwords):
        if word in stopwords:
            is_stopwords_flag = True
        else:
            is_stopwords_flag = False
        return is_stopwords_flag

    def _head_tail_is_stopwords(span, stopwords):
        if span[0] in stopwords or span[-1] in stopwords:
            return True
        else:
            return False

    def _with_septoken(ngram, tokenizer):
        if tokenizer.bos_token in ngram or tokenizer.sep_token in ngram or tokenizer.eos_token in ngram:
            flag = True
        else: flag = False
        return flag

    def _is_argument_words(seq, argument_words):
        pattern = None
        arg_words = list(argument_words.keys())
        if seq.strip() in arg_words:
            pattern = argument_words[seq.strip()]
        return pattern

    def _is_exist(exists:list, start:int, end:int):
        flag = False
        for estart, eend in exists:
            if estart <= start and eend >= end:
                flag = True
                break
        return flag

    def _find_punct(tokens, punctuations):
        punct_ids = [0] * len(tokens)
        for i, token in enumerate(tokens):
            if token in punctuations:
                punct_ids[i] = 8
            if token and token[-1] in [".","!","?"]:
                punct_ids[i] = 9
        return punct_ids

    def _find_arg_ngrams(tokens, max_gram, relations):
        n_tokens = len(tokens)
        global_arg_start_end = []
        argument_words = {}
        argument_ids = [0] * n_tokens
        for n in range(max_gram, 0, -1):  # loop over n-gram.
            for i in range(n_tokens - n):  # n-gram window sliding.
                window_start, window_end = i, i + n
                ngram = " ".join(tokens[window_start:window_end]).strip()
                pattern = _is_argument_words(ngram, relations)
                if pattern:
                    if not _is_exist(global_arg_start_end, window_start, window_end):
                        global_arg_start_end.append((window_start, window_end))
                        argument_ids[window_start:window_end] = [pattern] * (window_end - window_start)
                        argument_words[ngram] = (window_start, window_end)

        return argument_words, argument_ids

    def _find_dom_ngrams_2(tokens, max_gram):
        '''
        1. 判断 stopwords 和 sep token
        2. 先遍历一遍，记录 n-gram 的重复次数和出现位置
        3. 遍历记录的 n-gram, 过滤掉 n-gram 子序列（直接比较 str）
        4. 赋值 domain_ids.

        '''

        stemmed_tokens = [token_stem(token) for token in tokens]

        ''' 1 & 2'''
        n_tokens = len(tokens)
        d_ngram = {}
        domain_words_stemmed = {}
        domain_words_orin = {}
        domain_ids = [0] * n_tokens
        for n in range(max_gram, 0, -1):  # loop over n-gram.
            for i in range(n_tokens - n):  # n-gram window sliding.

                window_start, window_end = i, i+n
                stemmed_span = stemmed_tokens[window_start:window_end]
                stemmed_ngram = " ".join(stemmed_span)
                orin_span = tokens[window_start:window_end]
                orin_ngram = " ".join(orin_span)

                if _is_stopwords(orin_ngram, stopwords): continue
                if _head_tail_is_stopwords(orin_span, stopwords): continue
                if _with_septoken(orin_ngram, tokenizer): continue

                if not stemmed_ngram in d_ngram:
                    d_ngram[stemmed_ngram] = []
                d_ngram[stemmed_ngram].append((window_start, window_end))

        ''' 3 '''
        d_ngram = dict(filter(lambda e: len(e[1]) > 1, d_ngram.items()))
        raw_domain_words = list(d_ngram.keys())
        raw_domain_words.sort(key=lambda s: len(s), reverse=True)  # sort by len(str).
        domain_words_to_remove = []
        for i in range(0, len(d_ngram)):
            for j in range(i+1, len(d_ngram)):
                if raw_domain_words[i] in raw_domain_words[j]:
                    domain_words_to_remove.append(raw_domain_words[i])
                if raw_domain_words[j] in raw_domain_words[i]:
                    domain_words_to_remove.append(raw_domain_words[j])
        for r in domain_words_to_remove:
            try:
                del d_ngram[r]
            except:
                pass

        ''' 4 '''
        d_id = 0
        for stemmed_ngram, start_end_list in d_ngram.items():
            d_id += 1
            for start, end in start_end_list:
                domain_ids[start:end] = [d_id] * (end - start)
                rebuilt_orin_ngram = " ".join(tokens[start: end])
                if not stemmed_ngram in domain_words_stemmed:
                    domain_words_stemmed[stemmed_ngram] = []
                if not rebuilt_orin_ngram in domain_words_orin:
                    domain_words_orin[rebuilt_orin_ngram] = []
                domain_words_stemmed[stemmed_ngram] +=  [(start, end)]
                domain_words_orin[rebuilt_orin_ngram] += [(start, end)]


        return domain_words_stemmed, domain_words_orin, domain_ids

    def split_into_sentences(token_list, id_list):
        """
            function: split the context by the punctuations into sentences
            input: token_list, id_list
            output: sentence_list   List[List]
        """
        split_id_indices = np.where(np.array(id_list) == 9)[0].tolist()
        for i in range(len(split_id_indices)):
            if token_list[split_id_indices[i]+1] == '</s>':
                split_id_indices[i] += 1


        sentence_list = []
        for i in range(len(split_id_indices)):
            if i==0:
                sentence_list.append(token_list[0:split_id_indices[i]+1])
            else:
                if split_id_indices[i-1]+1 != split_id_indices[i]:   # address the '. . .' at the end of the sentence
                    sentence_list.append(token_list[split_id_indices[i-1]+1:split_id_indices[i]+1])
        return sentence_list, split_id_indices

    def check_variables(var_list):
        """
            function: check variables from the sentence
            input: sentence, space_id
            output: adjusted variable_list
        """
        variable_list = []
        for var in var_list:
            variable = var
            if len(var) != 0:
                if var[0] in ['.',',','?','!',':',';']:
                    variable = var[1:]
                if var[-1] in ['.',',','?','!',':',';']:
                    variable = var[:-1]
            variable_list.append(variable)
        return variable_list

    def get_atom(trigger_id, trigger_argument, sentence, space_id, space_ids_list):
        """
            function: convert the sentence into the atom form
                [ ( <id>, <trigger> ), [ [<variable>], [<variable>], [<variable>] ] ]
            input: trigger_id, trigger_argument, sentence, space_id, space_ids_list
            output: atom form
        """
        inversed = 0
        split_variable_ids = []   # relative position of the variables in the sub-sentence
        grouped_split_ids = []    # relative position of grouped variables and predicates in the sub-sentence
        vp_order = []     # the order tags of variables and predicates in the sub-sentence, 0: variable, 1: predicate
        atom = [(trigger_id, trigger_argument),]
        if trigger_id == 1:   # cause -->  result
            if 0 in space_ids_list[:space_id[0]] and 0 in space_ids_list[space_id[1]:]:   # trigger is in the middle
                if trigger_argument in ["because", "since", "due to", "because of"]:    # result + trigger + cause
                    var_list = [sentence[space_id[1]:], sentence[space_id[0]:space_id[1]], sentence[0:space_id[0]]]
                    inversed = 1
                else:       # cause + trigger + result
                    var_list = [sentence[0:space_id[0]], sentence[space_id[0]:space_id[1]], sentence[space_id[1]:]]
                split_variable_ids.append(space_id[0]-1)
                split_variable_ids.append(len(space_ids_list)-1)

            elif 0 in space_ids_list[space_id[1]:]:  # trigger in the front
                if trigger_argument in ["there fore", "t there fore", "thus", "so", "hence", "as a result", "consequently"]:   # only result, no cause
                    var_list = [[], [trigger_argument], sentence[space_id[1]:]]
                    split_variable_ids.append(len(space_ids_list)-1)
                else:    # trigger + cause + , + result
                    comma_id = None
                    for i in range(len(sentence)):
                        if i>=space_id[1] and sentence[i]==',':   # find the closest ',' in the sentence
                            comma_id = i
                    if comma_id:
                        var_list = [sentence[space_id[1]:comma_id], [trigger_argument], sentence[comma_id+1:]]
                        split_variable_ids.append(comma_id)
                        split_variable_ids.append(len(space_ids_list)-1)
                    else:
                        var_list = [[], [trigger_argument], sentence[space_id[1]:]]
                        split_variable_ids.append(len(space_ids_list)-1)
            else:   # trigger in the back
                var_list = [sentence]
                split_variable_ids.append(len(space_ids_list)-1)
                    
            atom.append(check_variables(var_list))

        elif trigger_id == 2:  # premise --->  hypothesis  (only if)
            if 0 in space_ids_list[:space_id[0]] and 0 in space_ids_list[space_id[1]:]:   # trigger is not in the front
                if trigger_argument == "unless" and "not" in sentence[0:space_id[0]]: #process "unless" and negation
                    adjust_hypo = sentence[0:space_id[0]].copy()
                    adjust_hypo.remove("not")
                    var_list = [sentence[space_id[1]:], sentence[space_id[0]:space_id[1]], adjust_hypo]
                else:
                    var_list = [sentence[space_id[1]:], sentence[space_id[0]:space_id[1]], sentence[0:space_id[0]]]
                inversed = 1
                split_variable_ids.append(space_id[0]-1)
                split_variable_ids.append(len(space_ids_list)-1)

            elif 0 in space_ids_list[space_id[1]:]:   # trigger in the front
                comma_id = None
                for i in range(len(sentence)):
                    if i>=space_id[1] and sentence[i]==',':   # find the closest ',' in the sentence
                        comma_id = i
                if comma_id:
                    if trigger_argument == "unless" and "not" in sentence[0:space_id[0]]:#process "unless" and negation
                        adjust_hypo = sentence[comma_id+1:].copy()
                        adjust_hypo.remove("not")
                        var_list = [sentence[space_id[1]:comma_id], [trigger_argument], adjust_hypo]
                    else:
                        var_list = [sentence[space_id[1]:comma_id], [trigger_argument], sentence[comma_id+1:]]
                    split_variable_ids.append(comma_id)
                    split_variable_ids.append(len(space_ids_list)-1)
                else:
                    var_list = [[], [trigger_argument], sentence[space_id[1]:]]
                    split_variable_ids.append(len(space_ids_list)-1)
            else:
                var_list = [sentence]
                split_variable_ids.append(len(space_ids_list)-1)
            atom.append(check_variables(var_list))
            
        elif trigger_id == 3:  # premise --->  hypothesis   (if)
            if 0 in space_ids_list[:space_id[0]] and 0 in space_ids_list[space_id[1]:]:   # trigger is not in the front
                if trigger_argument in ['if', 'once', 'as long as', 'as soon as']:
                    var_list = [sentence[space_id[1]:], sentence[space_id[0]:space_id[1]], sentence[0:space_id[0]]]
                    inversed = 1
                else:
                    var_list = [sentence[0:space_id[0]], sentence[space_id[0]:space_id[1]], sentence[space_id[1]:]]
                split_variable_ids.append(space_id[0]-1)
                split_variable_ids.append(len(space_ids_list)-1)
            elif 0 in space_ids_list[space_id[1]:]:   # trigger in the front or in the back
                if trigger_argument in ['if', 'once', 'as long as', 'as soon as']:
                    comma_id = None
                    for i in range(len(sentence)):
                        if i>=space_id[1] and sentence[i]==',':   # find the closest ',' in the sentence
                            comma_id = i
                    if comma_id:
                        var_list = [sentence[space_id[1]:comma_id], [trigger_argument], sentence[comma_id+1:]]
                        split_variable_ids.append(comma_id)
                        split_variable_ids.append(len(space_ids_list)-1)
                    else:
                        var_list = [[], [trigger_argument], sentence[space_id[1]:]]
                        split_variable_ids.append(len(space_ids_list)-1)
                else:
                    print(trigger_argument)
            else:
                var_list = [sentence]
                split_variable_ids.append(len(space_ids_list)-1)
            atom.append(check_variables(var_list))

            
        else:
            var_list = [sentence]
            atom.append(check_variables(var_list))
            split_variable_ids.append(len(space_ids_list)-1)
        return atom, split_variable_ids, inversed, grouped_split_ids, vp_order


    def has_same_logical_component(set1, set2):
        has_same = False
        overlap = -1
        if len(set1) > 1 and len(set2) > 1:
            overlap = len(set1 & set2)/max(min(len(set1), len(set2)), 1)
            if overlap >= 0.6:  # hyper-parameter:0.5
                has_same = True
        return has_same, overlap

    def tag_variables(tags, i, j, sents_list, max_tag, map_dict):
        """
            function: tag variables
            input: tag_list, current position i, j, sent_set, max_tag, map_dict
            output: tagged list
        """
        flag = False   # has same or not
        max_tag += 1
        current_sent = set(sents_list[i][j])-set(stopwords)
        for m in range(i+1, len(tags)):
            for n in range(len(tags[m])):
                comp_sent = set(sents_list[m][n])-set(stopwords)
                has_same,_ = has_same_logical_component(current_sent, comp_sent)
                if has_same:   # same variable
                    if tags[m][n] == -1:
                        tags[i][j] = max_tag
                        tags[m][n] = max_tag
                        flag = True
                        break
        if not flag:
            tags[i][j] = max_tag
            
        if tags[i][j] in map_dict.keys():     # write into the variable_text_map_dict
            map_dict[tags[i][j]].append(sent_list[i][j])
        else:
            map_dict[tags[i][j]] = sent_list[i][j]
        
        if flag:
            return tags,max_tag,map_dict
        else:
            return tags, max_tag,map_dict



    ''' start '''
    bpe_tokens_a = tokenizer.tokenize(text_a)
    bpe_tokens_b = tokenizer.tokenize(text_b)
    bpe_tokens_c = tokenizer.tokenize(text_c)

    # bpe_tokens_a = [token[1:].lower() if "Ġ" in token else token.lower() for token in bpe_tokens_a]
    # bpe_tokens_a = bpe_tokens_a[1:] if bpe_tokens_a[0]=='.' else bpe_tokens_a
    # bpe_tokens_b = [token[1:].lower() if "Ġ" in token else token.lower() for token in bpe_tokens_b]
    # bpe_tokens_b = bpe_tokens_b[1:] if bpe_tokens_b[0]=='.' else bpe_tokens_b
    # bpe_tokens_c = [token[1:].lower() if "Ġ" in token else token.lower() for token in bpe_tokens_c]
    # bpe_tokens_c = bpe_tokens_c[1:] if bpe_tokens_c[0]=='.' else bpe_tokens_c

    bpe_tokens = [tokenizer.bos_token] + bpe_tokens_a + [tokenizer.sep_token] + \
                    bpe_tokens_b + [tokenizer.sep_token] + \
                    bpe_tokens_c + [tokenizer.eos_token]

    a_mask = [1] * (len(bpe_tokens_a) + 2) + [0] * (max_length - (len(bpe_tokens_a) + 2))
    b_mask = [0] * (len(bpe_tokens_a) + 2) + [1] * (len(bpe_tokens_b) + 1) + [0] * (max_length - (len(bpe_tokens_a) + 2 + len(bpe_tokens_b) + 1))
    c_mask = [0] * (len(bpe_tokens_a) + 2) + [0] * (len(bpe_tokens_b) + 1) + [1] * (len(bpe_tokens_c) + 1) + [0] * (max_length - len(bpe_tokens))

    a_mask = a_mask[:max_length]
    b_mask = b_mask[:max_length]
    c_mask = c_mask[:max_length]
    assert len(a_mask) == max_length, 'len_a_mask={}, max_len={}'.format(len(a_mask), max_length)
    assert len(b_mask) == max_length, 'len_b_mask={}, max_len={}'.format(len(b_mask), max_length)
    assert len(c_mask) == max_length, 'len_c_mask={}, max_len={}'.format(len(c_mask), max_length)
    
    if len(bpe_tokens)>max_length:
        bpe_tokens = bpe_tokens[:max_length]
    if bpe_tokens[-1] != tokenizer.eos_token:
        bpe_tokens[-1] = tokenizer.eos_token
    # adapting Ġ.
    assert isinstance(bpe_tokens, list)
    bare_tokens = [token[1:].lower() if "Ġ" in token else token.lower() for token in bpe_tokens]
    # print(len(bare_tokens))


    """ get the atom forms for each instance """
    argument_words, argument_space_ids = _find_arg_ngrams(bare_tokens, max_gram=5, relations=relations)
    punct_space_ids = _find_punct(bare_tokens, punctuations)
    space_ids = [a+b for a,b in zip(argument_space_ids,punct_space_ids)]
    sentence_list, split_sentence_ids = split_into_sentences(bare_tokens, space_ids)
    argument_bpe_ids = argument_space_ids
    punct_bpe_ids = punct_space_ids
    space_bpe_ids = space_ids

    split_sentence_ids = [-1] + split_sentence_ids
    atom_sent = []
    atom_text_dict = {}
    split_bpe_ids, inverse_tags, grouped_split_ids, vp_order = [], [], [], []

    for index, s in enumerate(sentence_list):
        inversed = 0   # default: the atom will not inverse the order of the variables
        argument_words, argument_space_ids = _find_arg_ngrams(s, 5, relations)
        punct_space_ids = _find_punct(s, punctuations)
        space_ids = [a+b for a,b in zip(argument_space_ids,punct_space_ids)]
        if argument_words:
            argument_words_list = list(argument_words.keys())
            argument_words_list = [w.strip() for w in argument_words_list]
            argument_ids_list = [relations[argu] for argu in argument_words_list]

            trigger_index = argument_ids_list.index(min(argument_ids_list))
            trigger_argument = argument_words_list[trigger_index]
            trigger_id = relations[trigger_argument]  # the id number of the trigger in the dictionary
            atom, split_var_id, inversed, _, _ = get_atom(trigger_id, trigger_argument, s, argument_words[trigger_argument], space_ids)
            atom_sent.append(atom)  # get atom forms
            atom_text_dict[index] = s
        else:
            atom_sent.append([(7,"fact"), [s]])   # get atom forms
            atom_text_dict[index] = s
            split_var_id = [len(space_ids)-1]

        split_bpe_ids += [split_sentence_ids[index]+1+i for i in split_var_id]
        # print(split_bpe_ids)
        inverse_tags.append(inversed)

    # for i in range(len(split_bpe_ids)-1):
    #     if split_bpe_ids[i+1]-split_bpe_ids[i] <= 1:
    #         print(bare_tokens)
    #         print(split_bpe_ids)
    #         print("=========================")
    # print(split_bpe_ids)
    # print(bare_tokens)

    
    """ tag the same variables """
    tags, sent_list, negation_tag, predicate_list = [], [], [], []
    for sent in atom_sent:
        ''' generate the predicate for each sent and append into list '''
        predicate_list.append(sent[0][0])

        if len(sent[1])==1:
            tags.append([-1])
            sent_list.append(sent[1])
                
        elif len(sent[1])==3:
            if len(sent[1][0]) != 0:
                tags.append([-1,-1])
                sent_list.append([sent[1][0], sent[1][2]])
            else:
                tags.append([-1])
                sent_list.append([sent[1][2]])
        
        ''' identify the negation word '''
        neg_list = []
        for index, s in enumerate(sent[1]):
            if len(s) != 0 and index != 1:
                if "not" in s:
                    neg_list.append(1)
                else:
                    neg_list.append(0)
        negation_tag.append(neg_list)
            
    max_tag = -1
    map_dict = {}
    for i in range(len(tags)):
        for j in range(len(tags[i])):
            if tags[i][j] == -1:
                tags, max_tag, map_dict = tag_variables(tags,i,j,sent_list,max_tag,map_dict)   # return the tagged list

    # test_tag = []
    # for i in tags:
    #     test_tag += i
    # test_tag = [1 if i!=-1 else 0 for i in test_tag]

    # if len(split_bpe_ids)!=sum(test_tag):
    #     print(sent_list)
    #     print(split_bpe_ids)
    #     print(predicate_list)


    variable_tags, negation_tags = [], []
    for var_tag, neg_tag in zip(tags, negation_tag):
        if len(var_tag)==1:
            var_tag += [-1]
        if len(neg_tag)==1:
            neg_tag += [-1]
        variable_tags.append(var_tag)
        negation_tags.append(neg_tag)

    if len(variable_tags) == 0:
        print(bare_tokens)


    assert len(tags) == len(predicate_list) == len(negation_tags) == len(inverse_tags), 'len_tags={}, len_predicate={}, len_negation={},  \
                                                    len_inverse={}'.format(len(tags), len(predicate_list), len(negation_tags), len(inverse_tags))

    ''' output items '''
    max_tag_length = 100
    max_split_length = 50
    input_ids = tokenizer.convert_tokens_to_ids(bpe_tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * (len(bpe_tokens_a) + 2) + [1] * (len(bpe_tokens_b) + 1)
    padding = [0] * (max_length - len(input_ids))
    padding_ids = [tokenizer.pad_token_id] * (max_length - len(input_ids))
    split_bpe_padding_ids = [-1] * (max_split_length - len(split_bpe_ids))
    arg_dom_padding_ids = [-1] * (max_length - len(input_ids))
    pred_tag_padding_ids = [-1] * (max_tag_length - len(tags))
    tag_padding_ids = [[-1,-1]] * (max_tag_length - len(tags))

    no_contain = 0
    if split_bpe_ids != sorted(split_bpe_ids, reverse=False):
        no_contain = 1
        print("==========")
        print(split_bpe_ids)
        print(bare_tokens)
        
    input_ids += padding_ids
    argument_bpe_ids += arg_dom_padding_ids
    punct_bpe_ids += arg_dom_padding_ids
    space_bpe_ids += arg_dom_padding_ids
    split_bpe_ids += split_bpe_padding_ids
    predicate_list += pred_tag_padding_ids
    variable_tags += tag_padding_ids
    negation_tags += tag_padding_ids
    inverse_tags += pred_tag_padding_ids
    input_mask += padding
    segment_ids += padding

    if len(input_ids) > max_length:
        print(len(input_ids))
    input_ids = input_ids[:max_length]
    input_mask = input_mask[:max_length]
    segment_ids = segment_ids[:max_length]
    argument_bpe_ids = argument_bpe_ids[:max_length]
    punct_bpe_ids = punct_bpe_ids[:max_length]
    space_bpe_ids = space_bpe_ids[:max_length]
    split_bpe_ids = split_bpe_ids[:max_split_length]

    assert len(input_ids) <= max_length, 'len_input_ids={}, max_length={}'.format(len(input_ids), max_length)
    assert len(input_mask) <= max_length, 'len_input_mask={}, max_length={}'.format(len(input_mask), max_length)
    assert len(segment_ids) <= max_length, 'len_segment_ids={}, max_length={}'.format(len(segment_ids), max_length)
    assert len(argument_bpe_ids) <= max_length, 'len_argument_bpe_ids={}, max_length={}'.format(
        len(argument_bpe_ids), max_length)
    assert len(punct_bpe_ids) <= max_length, 'len_punct_bpe_ids={}, max_length={}'.format(
        len(punct_bpe_ids), max_length)
    assert len(space_bpe_ids) <= max_length, 'len_space_bpe_ids={}, max_length={}'.format(
        len(space_bpe_ids), max_length)
    assert len(split_bpe_ids) <= max_split_length, 'len_split_bpe_ids={}, max_split_length={}'.format(
        len(split_bpe_ids), max_split_length)
    assert len(predicate_list) <= max_tag_length, 'len_predicate_list={}, max_tag_length={}'.format(
        len(predicate_list), max_tag_length)
    assert len(variable_tags) <= max_tag_length, 'len_variable_tags={}, max_tag_length={}'.format(
        len(variable_tags), max_tag_length)
    assert len(negation_tags) <= max_tag_length, 'len_negation_tags={}, max_tag_length={}'.format(
        len(negation_tags), max_tag_length)
    assert len(inverse_tags) <= max_tag_length, 'len_inverse_tags={}, max_tag_length={}'.format(
        len(inverse_tags), max_tag_length)


    output = {}
    output["input_tokens"] = bpe_tokens
    output["input_ids"] = input_ids
    output["attention_mask"] = input_mask
    output["token_type_ids"] = segment_ids
    output["argument_bpe_ids"] = argument_bpe_ids
    output["punct_bpe_ids"] = punct_bpe_ids
    output["space_bpe_ids"] = space_bpe_ids
    output["split_bpe_ids"] = split_bpe_ids
    output["variable_tags"] = variable_tags
    output["predicate_tags"] = predicate_list
    output["negation_tags"] = negation_tags
    output["inverse_tags"] = inverse_tags
    output["a_mask"] = a_mask
    output["b_mask"] = b_mask
    output["c_mask"] = c_mask
    return output, no_contain


def main(text, option, question, logic, punctuations):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    stopwords = list(gensim.parsing.preprocessing.STOPWORDS) + punctuations

    inputs = arg_tokenizer(text, option, question, tokenizer, stopwords, relations, punctuations, 5, 256)

    ''' print '''
    # p = []
    # for token, arg, dom, pun in zip(inputs["input_tokens"], inputs["argument_bpe_ids"], inputs["domain_bpe_ids"],
    #                                 inputs["punct_bpe_ids"]):
    #     p.append((token, arg, dom, pun))
    # print(p)
    # print('input_tokens\n{}'.format(inputs["input_tokens"]))
    # print('input_ids\n{}, size={}'.format(inputs["input_ids"], len(inputs["input_ids"])))
    # print('attention_mask\n{}'.format(inputs["attention_mask"]))
    # print('token_type_ids\n{}'.format(inputs["token_type_ids"]))
    # print('argument_bpe_ids\n{}'.format(inputs["argument_bpe_ids"]))
    # print('domain_bpe_ids\n{}, size={}'.format(inputs["domain_bpe_ids"], len(inputs["domain_bpe_ids"])))
    # print('punct_bpe_ids\n{}'.format(inputs["punct_bpe_ids"]))


if __name__ == '__main__':

    import json
    from graph_building_blocks.argument_set_punctuation_v4 import punctuations
    with open('./graph_building_blocks/explicit_arg_set_v4.json', 'r') as f:
        relations = json.load(f)  # key: relations, value: ignore

    context = "There will be three sprinting projects in an institution's track and field sports, namely 60M, 100M and 200M. Lao Zhang, Lao Wang and Lao Li each participated in one of them, and the three people participated in different projects. Lao Li did not participate in 100M, Lao Wang participated in 60M.Xiao Li? Lao Zhang did not participate in 60M, Lao Wang participated in 200M."

    option = "I must be stupid because all intelligent people are nearsighted and I have perfect eyesight."
    question = "The pattern of reasoning displayed above most closely parallels which of the following?"

    logic = [[0,1], [2,0], [3,2], [4,1]]

    main(context, option, question, logic, punctuations)

