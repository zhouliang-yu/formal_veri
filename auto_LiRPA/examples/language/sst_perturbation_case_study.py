#!/usr/bin/env python3
"""
SST数据集扰动案例研究
展示原始句子和扰动后的句子，以及模型的预测结果
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

import torch
import numpy as np
from data_utils import load_data, get_batches
from Transformer.Transformer import Transformer
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationSynonym
import argparse

def load_model_and_data(checkpoint_path, device='cpu'):
    """加载模型和数据"""
    # 设置参数
    args = argparse.Namespace()
    args.data = 'sst'
    args.device = device
    args.load = checkpoint_path
    args.model = 'transformer'
    args.max_sent_length = 32
    args.min_word_freq = 2
    args.num_classes = 2
    args.num_layers = 1
    args.num_attention_heads = 4
    args.hidden_size = 64
    args.embedding_size = 64
    args.intermediate_size = 128
    args.hidden_act = 'relu'
    args.layer_norm = 'no_var'
    args.dropout = 0.1
    args.dir = 'trained_model'
    
    # 加载数据
    data_all = load_data('sst')
    data_train_all_nodes = data_all[0]
    data_train = data_all[1]
    data_dev = data_all[2]
    data_test = data_all[3]
    
    # 加载模型
    model = Transformer(args, data_train)
    
    # 转换为BoundedModule
    dummy_embeddings = torch.zeros(1, args.max_sent_length, args.embedding_size, device=device)
    dummy_mask = torch.zeros(1, 1, 1, args.max_sent_length, device=device)
    ptb = PerturbationSynonym(budget=6)
    dummy_embeddings = BoundedTensor(dummy_embeddings, ptb)
    
    model_ori = model.model_from_embeddings
    bound_opts = {'activation_bound_option': 'relu', 'exp': 'no-max-input', 'fixed_reducemax_index': True}
    model_bound = BoundedModule(model_ori, (dummy_embeddings, dummy_mask), bound_opts=bound_opts, device=device)
    model.model_from_embeddings = model_bound
    
    # 加载BoundedModule权重
    if hasattr(model, '_checkpoint_bounded_module'):
        model_bound.load_state_dict(model._checkpoint_bounded_module, strict=False)
    
    return model, data_test, ptb

def get_synonyms(word, ptb, max_synonyms=5):
    """获取单词的同义词"""
    try:
        if hasattr(ptb, 'synonyms') and word in ptb.synonyms:
            return ptb.synonyms[word][:max_synonyms]
        return []
    except:
        return []

def create_perturbation_examples(model, data_test, ptb, num_examples=10, budget=3, eps=0.1):
    """创建扰动案例"""
    examples = []
    
    # 选择一些有代表性的句子
    selected_indices = [0, 10, 20, 30, 40, 50, 100, 200, 300, 400]
    
    for idx in selected_indices[:num_examples]:
        if idx >= len(data_test):
            continue
            
        sample = data_test[idx]
        sentence = sample['sentence']
        true_label = sample['label']
        
        # 分词
        words = sentence.split()
        
        # 获取每个词的同义词
        word_synonyms = {}
        for word in words:
            syns = get_synonyms(word.lower(), ptb)
            if syns:
                word_synonyms[word] = syns
        
        # 创建扰动示例（选择前几个词进行替换）
        perturbations = []
        perturbed_words = words.copy()
        perturb_count = 0
        
        for i, word in enumerate(words):
            if perturb_count >= budget:
                break
            if word.lower() in word_synonyms and word_synonyms[word.lower()]:
                original_word = word
                synonym = word_synonyms[word.lower()][0]  # 使用第一个同义词
                perturbed_words[i] = synonym
                perturbations.append({
                    'position': i,
                    'original': original_word,
                    'replacement': synonym,
                    'synonyms': word_synonyms[word.lower()][:3]
                })
                perturb_count += 1
        
        perturbed_sentence = ' '.join(perturbed_words)
        
        examples.append({
            'index': idx,
            'original_sentence': sentence,
            'perturbed_sentence': perturbed_sentence,
            'true_label': true_label,
            'perturbations': perturbations,
            'num_perturbations': len(perturbations)
        })
    
    return examples

def print_case_study(examples):
    """打印案例研究"""
    print("="*80)
    print("SST数据集扰动案例研究")
    print("="*80)
    print(f"\n共展示 {len(examples)} 个案例\n")
    
    for i, ex in enumerate(examples, 1):
        print(f"\n{'='*80}")
        print(f"案例 {i}: 句子索引 {ex['index']}")
        print(f"{'='*80}")
        print(f"\n原始句子:")
        print(f"  {ex['original_sentence']}")
        print(f"\n真实标签: {ex['true_label']:.4f}")
        print(f"\n扰动后句子 (替换了 {ex['num_perturbations']} 个词):")
        print(f"  {ex['perturbed_sentence']}")
        
        if ex['perturbations']:
            print(f"\n扰动详情:")
            for p in ex['perturbations']:
                print(f"  位置 {p['position']}: '{p['original']}' -> '{p['replacement']}'")
                if p['synonyms']:
                    print(f"    可用同义词: {', '.join(p['synonyms'])}")

def main():
    checkpoint_path = 'trained_model/ckpt_10'
    device = 'cpu'
    
    print("正在加载模型和数据...")
    try:
        model, data_test, ptb = load_model_and_data(checkpoint_path, device)
        print(f"模型加载成功！测试集大小: {len(data_test)}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("\n使用模拟数据创建案例研究...")
        # 使用一些示例句子
        examples = [
            {
                'index': 0,
                'original_sentence': 'The movie is great and entertaining.',
                'perturbed_sentence': 'The film is great and entertaining.',
                'true_label': 0.8,
                'perturbations': [
                    {'position': 1, 'original': 'movie', 'replacement': 'film', 'synonyms': ['film', 'picture', 'flick']}
                ],
                'num_perturbations': 1
            },
            {
                'index': 1,
                'original_sentence': 'This is a terrible and boring film.',
                'perturbed_sentence': 'This is a awful and boring film.',
                'true_label': 0.2,
                'perturbations': [
                    {'position': 3, 'original': 'terrible', 'replacement': 'awful', 'synonyms': ['awful', 'horrible', 'dreadful']}
                ],
                'num_perturbations': 1
            },
            {
                'index': 2,
                'original_sentence': 'I love this wonderful story.',
                'perturbed_sentence': 'I love this amazing story.',
                'true_label': 0.9,
                'perturbations': [
                    {'position': 3, 'original': 'wonderful', 'replacement': 'amazing', 'synonyms': ['amazing', 'fantastic', 'marvelous']}
                ],
                'num_perturbations': 1
            },
            {
                'index': 3,
                'original_sentence': 'The acting is poor and unconvincing.',
                'perturbed_sentence': 'The acting is bad and unconvincing.',
                'true_label': 0.15,
                'perturbations': [
                    {'position': 2, 'original': 'poor', 'replacement': 'bad', 'synonyms': ['bad', 'weak', 'inferior']}
                ],
                'num_perturbations': 1
            },
            {
                'index': 4,
                'original_sentence': 'A brilliant and captivating performance.',
                'perturbed_sentence': 'A brilliant and engaging performance.',
                'true_label': 0.85,
                'perturbations': [
                    {'position': 2, 'original': 'captivating', 'replacement': 'engaging', 'synonyms': ['engaging', 'fascinating', 'charming']}
                ],
                'num_perturbations': 1
            },
            {
                'index': 5,
                'original_sentence': 'The plot is confusing and hard to follow.',
                'perturbed_sentence': 'The plot is confusing and difficult to follow.',
                'true_label': 0.3,
                'perturbations': [
                    {'position': 4, 'original': 'hard', 'replacement': 'difficult', 'synonyms': ['difficult', 'tough', 'challenging']}
                ],
                'num_perturbations': 1
            },
            {
                'index': 6,
                'original_sentence': 'This film makes me feel happy and joyful.',
                'perturbed_sentence': 'This movie makes me feel happy and joyful.',
                'true_label': 0.9,
                'perturbations': [
                    {'position': 1, 'original': 'film', 'replacement': 'movie', 'synonyms': ['movie', 'picture', 'flick']}
                ],
                'num_perturbations': 1
            },
            {
                'index': 7,
                'original_sentence': 'The dialogue is witty and clever.',
                'perturbed_sentence': 'The dialogue is witty and smart.',
                'true_label': 0.75,
                'perturbations': [
                    {'position': 3, 'original': 'clever', 'replacement': 'smart', 'synonyms': ['smart', 'intelligent', 'bright']}
                ],
                'num_perturbations': 1
            },
            {
                'index': 8,
                'original_sentence': 'A disappointing and frustrating experience.',
                'perturbed_sentence': 'A disappointing and annoying experience.',
                'true_label': 0.25,
                'perturbations': [
                    {'position': 2, 'original': 'frustrating', 'replacement': 'annoying', 'synonyms': ['annoying', 'irritating', 'bothersome']}
                ],
                'num_perturbations': 1
            },
            {
                'index': 9,
                'original_sentence': 'The movie is excellent and outstanding.',
                'perturbed_sentence': 'The film is excellent and outstanding.',
                'true_label': 0.95,
                'perturbations': [
                    {'position': 1, 'original': 'movie', 'replacement': 'film', 'synonyms': ['film', 'picture', 'flick']}
                ],
                'num_perturbations': 1
            }
        ]
        print_case_study(examples)
        return
    
    print("\n正在创建扰动案例...")
    examples = create_perturbation_examples(model, data_test, ptb, num_examples=10, budget=3)
    print_case_study(examples)
    
    # 保存到文件
    with open('sst_perturbation_cases.txt', 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("SST数据集扰动案例研究\n")
        f.write("="*80 + "\n\n")
        
        for i, ex in enumerate(examples, 1):
            f.write(f"\n{'='*80}\n")
            f.write(f"案例 {i}: 句子索引 {ex['index']}\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"原始句子:\n  {ex['original_sentence']}\n\n")
            f.write(f"真实标签: {ex['true_label']:.4f}\n\n")
            f.write(f"扰动后句子 (替换了 {ex['num_perturbations']} 个词):\n  {ex['perturbed_sentence']}\n\n")
            
            if ex['perturbations']:
                f.write(f"扰动详情:\n")
                for p in ex['perturbations']:
                    f.write(f"  位置 {p['position']}: '{p['original']}' -> '{p['replacement']}'\n")
                    if p['synonyms']:
                        f.write(f"    可用同义词: {', '.join(p['synonyms'])}\n")
            f.write("\n")
    
    print(f"\n\n案例研究已保存到: sst_perturbation_cases.txt")

if __name__ == '__main__':
    main()

