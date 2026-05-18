# 正则化：梯度裁剪（Gradient Clipping）

**首次出现**：RNN  
**解决的问题**：RNN/LSTM 训练中常见的梯度爆炸问题

---

## 1. 问题背景：为什么 RNN 容易梯度爆炸？

RNN 在反向传播时，需要沿时间轴展开计算（BPTT），梯度通过连乘传递：

$$\frac{\partial L}{\partial h_1} \propto \prod_{t=2}^{T} W_{hh} \cdot \tanh'(h_t)$$

当序列长度 $T$ 很大时：

- 若 $\|W_{hh}\| > 1$：梯度指数级增大 → **梯度爆炸**
- 若 $\|W_{hh}\| < 1$：梯度指数级衰减 → **梯度消失**

梯度消失需要 LSTM 等架构来根治；梯度爆炸可以用**梯度裁剪**快速解决。

---

## 2. 梯度裁剪原理

将所有参数梯度拼接为一个大向量 $\mathbf{g}$，计算其 L2 范数：

$$\|\mathbf{g}\|_2 = \sqrt{\sum_i g_i^2}$$

若范数超过阈值 $\text{max\_norm}$，等比例缩小所有梯度：

$$\mathbf{g} \leftarrow \mathbf{g} \cdot \frac{\text{max\_norm}}{\|\mathbf{g}\|_2} \quad \text{当 } \|\mathbf{g}\|_2 > \text{max\_norm}$$

**关键性质**：
- 梯度方向不变（所有分量等比缩放）
- 只限制梯度"有多大"，不改变更新方向
- 阈值内不做任何修改（不是 L1/L2 正则化那样的持续惩罚）

---

## 3. PyTorch 使用

```python
# 标准用法：在 loss.backward() 后、optimizer.step() 前调用
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
optimizer.step()
```

**参数说明**：

| 参数 | 含义 |
|------|------|
| `parameters` | 需要裁剪的参数组（通常 `model.parameters()`） |
| `max_norm` | L2 范数上限（常用值：1.0 / 5.0 / 10.0） |
| `norm_type` | 范数类型（默认 2.0，即 L2 范数） |

**返回值**：裁剪前的梯度范数（可用于监控）：
```python
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
print(f"梯度范数: {grad_norm:.4f}")
```

---

## 4. 按值裁剪（Clip by Value）

另一种裁剪方式：将每个梯度值独立限制在 $[-c, c]$：

```python
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
```

**对比**：

| | clip_grad_norm_ | clip_grad_value_ |
|--|----------------|-----------------|
| 操作 | 缩放整体向量 | 逐元素截断 |
| 保留梯度方向 | ✓ | ✗（各分量独立截断） |
| 使用场景 | RNN/LSTM（推荐） | 简单场景 |

---

## 5. max_norm 如何选取？

经验规律：

| 场景 | 推荐值 |
|------|-------|
| LSTM/RNN 文本分类 | 1.0 ~ 5.0 |
| RNN 语言模型 | 0.25 ~ 1.0 |
| Transformer | 1.0（部分实现不用裁剪） |
| 通用默认 | 5.0 |

**调参方法**：在训练前几个 epoch 打印梯度范数，选择 max_norm 略大于典型值：

```python
# 监控梯度范数而不实际裁剪
total_norm = 0
for p in model.parameters():
    if p.grad is not None:
        total_norm += p.grad.data.norm(2).item() ** 2
total_norm = total_norm ** 0.5
print(f"梯度 L2 范数: {total_norm:.4f}")
```

---

## 6. 常见误用

| 误用 | 后果 | 正确做法 |
|------|------|---------|
| 在 `loss.backward()` 之前调用 | 梯度还未计算，裁剪无效 | 必须在 backward() 之后 |
| 在 `optimizer.step()` 之后调用 | 参数已更新，裁剪无效 | 必须在 step() 之前 |
| max_norm 设置过小（如 0.01） | 梯度几乎全被截断，训练不收敛 | 先监控梯度范数再设值 |
| MLP/CNN 也加梯度裁剪 | 通常不必要，MLP/CNN 较少出现梯度爆炸 | 只在 RNN/LSTM 等序列模型中使用 |
