#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pickle


class Attns(object):
    """Visualize attention in HTML."""

    def __init__(self, attns_pkl_filename,):
        with open(attns_pkl_filename, "rb") as f:
            sentences = pickle.load(f)
        self.sentences = sentences

    def __get_elem(self, token, attn, max_attn):
        attn_int = int(attn * max_attn)
        print(token)
        token = token.replace(' ','')
        print(token)
        return ('<span style="background-color:#ff0000{attn_int:02x};">' + \
            '{token}' + \
            '</span>').format(**locals())

    def to_html(self, style='<style>th, td { text-align: left!important; white-space: nowrap!important; }</style>', label_format='Head{head_index}', max_attn=240):
        html_string = ''

        # 文書毎に計算
        for sentence_index, sentence in enumerate(self.sentences):
            sentence_text = sentence['text']
            sentence_class = sentence['class'] if 'class' in sentence else ''
            sentence_info1 = sentence['info1'] if 'info1' in sentence else ''
            sentence_info2 = sentence['info2'] if 'info2' in sentence else ''
            sentence_tokens = sentence['tokens']
            tokens_size = len(sentence['tokens'])
            layers_size = len(sentence['attns'])
            heads_size = len(sentence['attns'][0])
            
            # layer毎に計算
            html_string += '<h1 id="s{sentence_index}">Sentence{sentence_index} - {sentence_text}</h1>'.format(**locals())
            html_string += '<div>{sentence_class} {sentence_info1} {sentence_info2}</div><br/>'.format(**locals())
            for layer_index in range(layers_size):
                heads_data = {}

                # head毎のアテンションを計算
                for head_index in range(heads_size):
                    token_attns = np.zeros(tokens_size, dtype=np.float)
                    for col_index in range(tokens_size):
                        token_attn = 0.0
                        for row_index in range(tokens_size):
                            token_attn += sentence['attns'][layer_index][head_index][row_index][col_index]
                        token_attns[col_index] = token_attn
                    token_attns_min, token_attns_max = token_attns.min(), token_attns.max()
                    token_attns = (token_attns - token_attns_min) / (token_attns_max - token_attns_min)
                    label = label_format.format(**locals())
                    heads_data[label] = [ self.__get_elem(token, token_attns[token_index], max_attn) for token_index, token in enumerate(sentence_tokens) ]

                # head全体のアテンションを計算
                token_attns = np.zeros(tokens_size, dtype=np.float)
                for head_index in range(heads_size):
                    for col_index in range(tokens_size):
                        token_attn = 0.0
                        for row_index in range(tokens_size):
                            token_attn += sentence['attns'][layer_index][head_index][row_index][col_index]
                        token_attns[col_index] = token_attn
                token_attns_min, token_attns_max = token_attns.min(), token_attns.max()
                token_attns = (token_attns - token_attns_min) / (token_attns_max - token_attns_min)
                label = '全体'.format(**locals())
                heads_data[label] = [ self.__get_elem(token, token_attns[token_index], max_attn) for token_index, token in enumerate(sentence_tokens) ]
                
                # headの全結果をHTML化
                heads_df = pd.DataFrame(heads_data)
                heads_html = heads_df.T.to_html(escape=False, header=False)
                html_string += '<details open id="s{sentence_index}-l{layer_index}"><summary>layer{layer_index}</summary>{heads_html}</details>'.format(**locals())

        # 全文書の結果を返す
        return style + html_string


if __name__ == "__main__":
    import argparse
    import codecs

    parser = argparse.ArgumentParser()
    parser.add_argument('attns_pkl_filename')
    parser.add_argument('html_filename')
    args = parser.parse_args()

    html_string = Attns(args.attns_pkl_filename).to_html()

    with codecs.open(args.html_filename, 'w', 'utf-8') as fout:
        fout.write(html_string)
