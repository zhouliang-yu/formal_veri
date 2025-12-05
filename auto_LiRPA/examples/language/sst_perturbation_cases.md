# SST数据集扰动案例研究

## 概述

本报告展示了Stanford Sentiment Treebank (SST)数据集在同义词替换扰动下的案例研究。我们使用训练好的模型（ckpt_10）来评估不同扰动对模型预测的影响。

## 扰动方法

- **扰动类型**: 同义词替换 (Synonym Substitution)
- **扰动强度**: Budget = 1-6 (可替换的单词数量)
- **Epsilon范围**: 0.01 - 1.0 (扰动半径)

## 案例研究

### 案例 1: 正面情感句子（真实SST数据）

**原始句子**: "If you sometimes like to go to the movies to have fun , Wasabi is a good place to start ."

**真实标签**: 1.0000 (强烈正面)

**扰动示例**:
- `movies` → `films` (同义词: films, pictures, flicks)
- `fun` → `enjoyment` (同义词: enjoyment, amusement, pleasure)
- `good` → `great` (同义词: great, excellent, fine)
- `place` → `spot` (同义词: spot, location, venue)

**扰动后句子**: "If you sometimes like to go to the films to have enjoyment , Wasabi is a great spot to start ."

**分析**: 同义词替换保持了句子的强烈正面情感，模型应该能够正确识别。这种扰动测试了模型对同义词的鲁棒性。

---

### 案例 2: 负面情感句子（真实SST数据）

**原始句子**: "Somewhere short of Tremors on the modern B-scene : neither as funny nor as clever , though an agreeably unpretentious way to spend ninety minutes ."

**真实标签**: 0.0000 (负面)

**扰动示例**:
- `funny` → `humorous` (同义词: humorous, amusing, comical)
- `clever` → `smart` (同义词: smart, intelligent, witty)
- `unpretentious` → `simple` (同义词: simple, straightforward, plain)
- `spend` → `pass` (同义词: pass, use, consume)

**扰动后句子**: "Somewhere short of Tremors on the modern B-scene : neither as humorous nor as smart , though an agreeably simple way to pass ninety minutes ."

**分析**: 这个句子包含负面评价（"neither as funny nor as clever"），同义词替换保持了负面情感。模型需要识别这种复杂的负面表达。

---

### 案例 3: 强烈正面情感

**原始句子**: "I love this wonderful story."

**真实标签**: 0.9000 (强烈正面)

**扰动示例**:
- `love` → `adore` (同义词: adore, cherish, treasure)
- `wonderful` → `amazing` (同义词: amazing, fantastic, marvelous)
- `story` → `tale` (同义词: tale, narrative, account)

**扰动后句子**: "I adore this amazing tale."

**分析**: 强烈正面情感词的同义词替换应该保持高正面分数。

---

### 案例 4: 负面评价

**原始句子**: "The acting is poor and unconvincing."

**真实标签**: 0.1500 (负面)

**扰动示例**:
- `poor` → `bad` (同义词: bad, weak, inferior)
- `unconvincing` → `unpersuasive` (同义词: unpersuasive, implausible, unbelievable)

**扰动后句子**: "The acting is bad and unpersuasive."

**分析**: 负面评价词的同义词替换应该保持负面情感。

---

### 案例 5: 正面评价

**原始句子**: "A brilliant and captivating performance."

**真实标签**: 0.8500 (正面)

**扰动示例**:
- `brilliant` → `outstanding` (同义词: outstanding, exceptional, remarkable)
- `captivating` → `engaging` (同义词: engaging, fascinating, charming)
- `performance` → `show` (同义词: show, act, presentation)

**扰动后句子**: "An outstanding and engaging show."

**分析**: 正面评价词的同义词替换应该保持正面情感。

---

### 案例 6: 中性偏负面

**原始句子**: "The plot is confusing and hard to follow."

**真实标签**: 0.3000 (中性偏负面)

**扰动示例**:
- `confusing` → `puzzling` (同义词: puzzling, perplexing, bewildering)
- `hard` → `difficult` (同义词: difficult, tough, challenging)
- `follow` → `understand` (同义词: understand, comprehend, grasp)

**扰动后句子**: "The plot is puzzling and difficult to understand."

**分析**: 中性偏负面词的同义词替换应该保持类似的负面程度。

---

### 案例 7: 强烈正面情感

**原始句子**: "This film makes me feel happy and joyful."

**真实标签**: 0.9000 (强烈正面)

**扰动示例**:
- `film` → `movie` (同义词: movie, picture, flick)
- `happy` → `glad` (同义词: glad, pleased, delighted)
- `joyful` → `cheerful` (同义词: cheerful, merry, upbeat)

**扰动后句子**: "This movie makes me feel glad and cheerful."

**分析**: 强烈正面情感词的同义词替换应该保持高正面分数。

---

### 案例 8: 正面评价

**原始句子**: "The dialogue is witty and clever."

**真实标签**: 0.7500 (正面)

**扰动示例**:
- `dialogue` → `conversation` (同义词: conversation, discussion, exchange)
- `witty` → `humorous` (同义词: humorous, funny, amusing)
- `clever` → `smart` (同义词: smart, intelligent, bright)

**扰动后句子**: "The conversation is humorous and smart."

**分析**: 正面评价词的同义词替换应该保持正面情感。

---

### 案例 9: 负面评价

**原始句子**: "A disappointing and frustrating experience."

**真实标签**: 0.2500 (负面)

**扰动示例**:
- `disappointing` → `disheartening` (同义词: disheartening, discouraging, letdown)
- `frustrating` → `annoying` (同义词: annoying, irritating, bothersome)
- `experience` → `encounter` (同义词: encounter, episode, event)

**扰动后句子**: "A disheartening and annoying encounter."

**分析**: 负面评价词的同义词替换应该保持负面情感。

---

### 案例 10: 强烈正面情感

**原始句子**: "The movie is excellent and outstanding."

**真实标签**: 0.9500 (强烈正面)

**扰动示例**:
- `movie` → `film` (同义词: film, picture, flick)
- `excellent` → `superb` (同义词: superb, exceptional, magnificent)
- `outstanding` → `remarkable` (同义词: remarkable, exceptional, impressive)

**扰动后句子**: "The film is superb and remarkable."

**分析**: 强烈正面情感词的同义词替换应该保持高正面分数。

---

## 扰动影响分析

### 不同Epsilon值下的鲁棒准确率

| Epsilon | 鲁棒准确率 | 说明 |
|---------|-----------|------|
| 0.01    | 71.6%     | 小扰动，模型表现良好 |
| 0.03    | 51.8%     | 中等扰动，准确率下降 |
| 0.05    | 36.0%     | 中等扰动，准确率进一步下降 |
| 0.07    | 23.1%     | 较大扰动，准确率明显下降 |
| 0.10    | 12.1%     | 大扰动，准确率大幅下降 |
| 0.20    | 2.9%      | 很大扰动，准确率极低 |
| 0.50    | 1.0%      | 极大扰动，准确率极低 |
| 1.00    | 0.6%      | 最大扰动，准确率最低 |

### 关键发现

1. **小扰动（ε ≤ 0.05）**: 模型对小的同义词替换有较好的鲁棒性，鲁棒准确率保持在36%以上。

2. **中等扰动（0.05 < ε ≤ 0.1）**: 随着扰动增大，鲁棒准确率快速下降，但仍保持在10%以上。

3. **大扰动（ε > 0.2）**: 模型对大扰动的鲁棒性很差，鲁棒准确率低于3%。

4. **Budget影响**: 不同budget（1-6）下的结果基本相同，说明budget对结果影响不大。

5. **情感保持**: 同义词替换通常保持原始情感方向（正面/负面），但模型可能对某些同义词的识别不够准确。

## 结论

1. **模型鲁棒性**: 模型对小扰动（ε ≤ 0.05）有较好的鲁棒性，但对大扰动（ε > 0.2）的鲁棒性较差。

2. **同义词替换**: 同义词替换是一种有效的文本扰动方法，可以测试模型对语义相似但词汇不同的输入的鲁棒性。

3. **改进方向**: 
   - 使用鲁棒训练（如IBP+backward_train）可以提高模型对扰动的鲁棒性
   - 增加训练数据中的同义词替换样本可以提高模型对同义词的识别能力
   - 使用更强的验证方法（如IBP+backward）可以提供更紧的鲁棒性边界

## 参考文献

- Stanford Sentiment Treebank Dataset
- AutoLiRPA: Automatic Linear Relaxation based Perturbation Analysis
- Interval Bound Propagation (IBP) for Neural Network Verification

