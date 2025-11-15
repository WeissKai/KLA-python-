# KLA Python API 使用指南

## 📖 目录

1. [基础使用](#基础使用)
2. [标准KLA算法](#标准kla算法)
3. [Warm-start增强](#warm-start增强)
4. [自定义优化问题](#自定义优化问题)
5. [API参考](#api参考)

---

## 基础使用

### 安装和导入

```python
# 确保已安装依赖
# pip install -r requirements.txt

import sys
import os
# 如果不在项目根目录，需要添加路径
sys.path.insert(0, '/path/to/KLA')

from src import kla_optimize, cost, Solution
```

---

## 标准KLA算法

### 1. 最简单的使用

```python
from src import kla_optimize, cost

# 运行KLA算法优化测试函数1
best_sol, history = kla_optimize(
    cost_function=cost,
    n_var=30,           # 30维变量
    var_min=-100,       # 下界
    var_max=100,        # 上界
    func_num=1          # 测试函数编号
)

print(f"最优成本: {best_sol.cost}")
print(f"最优解位置: {best_sol.position}")
```

### 2. 自定义参数

```python
best_sol, history = kla_optimize(
    cost_function=cost,
    n_var=30,
    var_min=-100,
    var_max=100,
    max_it=5000,        # 最大迭代次数（默认3000）
    n_pop=100,          # 种群大小（默认50）
    func_num=2
)

# 查看收敛历史
import matplotlib.pyplot as plt
plt.plot(history)
plt.xlabel('迭代次数')
plt.ylabel('最优成本')
plt.yscale('log')
plt.show()
```

### 3. 多次运行对比

```python
import numpy as np

results = []
for run in range(10):
    np.random.seed(run)
    best_sol, _ = kla_optimize(
        cost_function=cost,
        n_var=30,
        var_min=-100,
        var_max=100,
        func_num=1
    )
    results.append(best_sol.cost)

print(f"平均成本: {np.mean(results)}")
print(f"最优成本: {np.min(results)}")
print(f"标准差: {np.std(results)}")
```

---

## Warm-start增强

### 1. 训练Surrogate模型

```python
from src.warmstart import MetaSurrogate, generate_meta_training_data

# 步骤1: 生成元训练数据
print("生成训练数据...")
D_meta = generate_meta_training_data(
    n_tasks=50,              # 任务数量
    n_samples_per_task=2000, # 每个任务的样本数
    n_var=30,
    var_min=-100,
    var_max=100
)

# 步骤2: 创建并训练surrogate模型
print("训练Surrogate模型...")
surrogate = MetaSurrogate(
    model_type='mlp',                    # 'mlp' 或 'rf' (随机森林)
    hidden_layers=(256, 128, 64, 32)     # MLP的隐藏层结构
)

surrogate.train(D_meta, normalize_y=True, verbose=True)

# 步骤3: 保存模型供后续使用
surrogate.save('models/my_surrogate.pkl')
```

### 2. 加载已训练的模型

```python
from src.warmstart import MetaSurrogate

surrogate = MetaSurrogate()
surrogate.load('models/my_surrogate.pkl')
```

### 3. 使用Warm-start运行KLA

```python
from src import kla_optimize, cost

best_sol, history = kla_optimize(
    cost_function=cost,
    n_var=30,
    var_min=-100,
    var_max=100,
    func_num=1,
    # Warm-start参数
    surrogate=surrogate,
    use_warm_start=True,
    warm_start_params={
        'n_cand': 2000,              # 候选点数量
        'alpha_mix': 0.5,            # 随机混合比例(0.5=50%随机+50%surrogate)
        'sampling_method': 'lhs',    # 'uniform', 'lhs', 'sobol'
        'diversity_threshold': None, # None表示自动计算
        'verbose': True              # 显示详细信息
    }
)
```

### 4. 完整的Warm-start工作流

```python
from src import kla_optimize, cost
from src.warmstart import MetaSurrogate, generate_meta_training_data
import numpy as np

# 1. 训练（只需一次）
try:
    surrogate = MetaSurrogate()
    surrogate.load('models/my_surrogate.pkl')
    print("加载已有模型")
except:
    print("训练新模型...")
    D_meta = generate_meta_training_data(n_tasks=50, n_samples_per_task=2000)
    surrogate = MetaSurrogate(model_type='mlp', hidden_layers=(256, 128, 64, 32))
    surrogate.train(D_meta, normalize_y=True)
    surrogate.save('models/my_surrogate.pkl')

# 2. 对比标准KLA和Warm-start KLA
results_std = []
results_ws = []

for run in range(5):
    np.random.seed(run)
    
    # 标准KLA
    best_std, _ = kla_optimize(
        cost_function=cost, n_var=30, var_min=-100, var_max=100,
        func_num=1, use_warm_start=False
    )
    results_std.append(best_std.cost)
    
    # Warm-start KLA
    best_ws, _ = kla_optimize(
        cost_function=cost, n_var=30, var_min=-100, var_max=100,
        func_num=1, surrogate=surrogate, use_warm_start=True,
        warm_start_params={'n_cand': 2000, 'alpha_mix': 0.5}
    )
    results_ws.append(best_ws.cost)

print(f"标准KLA平均: {np.mean(results_std):.6e}")
print(f"Warm-start平均: {np.mean(results_ws):.6e}")
print(f"改进: {(np.mean(results_std) - np.mean(results_ws))/np.mean(results_std)*100:.2f}%")
```

---

## 自定义优化问题

### 1. 定义自己的目标函数

```python
import numpy as np

def my_objective_function(x, problem_params=None):
    """
    自定义目标函数
    
    参数:
        x: numpy.ndarray, 形状 (n_samples, n_dimensions)
        problem_params: 可选的问题参数
    
    返回:
        y: numpy.ndarray, 形状 (n_samples,)
    """
    # 例如：Rosenbrock函数
    result = np.zeros(x.shape[0])
    for i in range(x.shape[1] - 1):
        result += 100 * (x[:, i+1] - x[:, i]**2)**2 + (1 - x[:, i])**2
    return result

# 使用自定义函数
from src import kla_optimize

best_sol, history = kla_optimize(
    cost_function=my_objective_function,
    n_var=10,
    var_min=-5,
    var_max=10,
    max_it=3000,
    n_pop=50,
    func_num=None  # 自定义函数时可以传None
)
```

### 2. 带参数的目标函数

```python
def parametric_function(x, params):
    """带参数的目标函数"""
    a, b, c = params['a'], params['b'], params['c']
    return a * np.sum(x**2, axis=1) + b * np.sum(x, axis=1) + c

# 创建包装函数
params = {'a': 2, 'b': -1, 'c': 10}
cost_func = lambda x, fn: parametric_function(x, params)

best_sol, _ = kla_optimize(
    cost_function=cost_func,
    n_var=20,
    var_min=-10,
    var_max=10
)
```

### 3. 约束优化（惩罚函数法）

```python
def constrained_objective(x, penalty_weight=1000):
    """
    带约束的目标函数
    使用惩罚函数法处理约束
    """
    # 原始目标函数
    f = np.sum(x**2, axis=1)
    
    # 约束1: x1 + x2 <= 5
    g1 = x[:, 0] + x[:, 1] - 5
    penalty1 = penalty_weight * np.maximum(0, g1)**2
    
    # 约束2: x1 >= 0
    g2 = -x[:, 0]
    penalty2 = penalty_weight * np.maximum(0, g2)**2
    
    return f + penalty1 + penalty2

best_sol, _ = kla_optimize(
    cost_function=lambda x, fn: constrained_objective(x),
    n_var=5,
    var_min=-10,
    var_max=10
)
```

---

## API参考

### `kla_optimize()`

**主要函数：运行KLA优化算法**

```python
def kla_optimize(
    cost_function,      # 目标函数
    n_var,              # 决策变量数量
    var_min,            # 变量下界
    var_max,            # 变量上界
    max_it=3000,        # 最大函数评估次数
    n_pop=50,           # 种群大小
    func_num=1,         # 测试函数编号
    surrogate=None,     # Surrogate模型（用于warm-start）
    use_warm_start=False,           # 是否使用warm-start
    warm_start_params=None          # Warm-start参数字典
)
```

**返回值:**
- `best_sol`: Solution对象，包含`.position`和`.cost`属性
- `best_cost_history`: numpy数组，记录每次迭代的最优成本

**示例:**
```python
best_sol, history = kla_optimize(
    cost_function=my_func,
    n_var=30,
    var_min=-100,
    var_max=100
)
```

### `MetaSurrogate`

**元学习代理模型类**

```python
from src.warmstart import MetaSurrogate

# 创建模型
surrogate = MetaSurrogate(
    model_type='mlp',              # 'mlp' 或 'rf'
    hidden_layers=(256, 128, 64),  # MLP隐藏层
    random_state=42                # 随机种子
)

# 训练模型
surrogate.train(D_meta, normalize_y=True, verbose=True)

# 预测
predictions = surrogate.predict(X_candidates)

# 保存/加载
surrogate.save('path/to/model.pkl')
surrogate.load('path/to/model.pkl')
```

### `generate_meta_training_data()`

**生成元训练数据**

```python
from src.warmstart import generate_meta_training_data

D_meta = generate_meta_training_data(
    n_tasks=50,              # 任务数量
    n_samples_per_task=2000, # 每个任务的样本数
    n_var=30,                # 变量维度
    var_min=-100,            # 下界
    var_max=100              # 上界
)
```

**返回值:**
- `D_meta`: 列表，每个元素是字典 `{'X': numpy.ndarray, 'y': numpy.ndarray}`

### `warm_start_initialization()`

**生成warm-start初始种群**

```python
from src.warmstart import warm_start_initialization

X_init = warm_start_initialization(
    surrogate=surrogate,
    search_space=(var_min, var_max, n_var),
    n_pop=50,
    n_cand=2000,
    alpha_mix=0.5,
    diversity_threshold=None,
    sampling_method='lhs',
    verbose=True
)
```

### `cost()`

**内置测试函数**

```python
from src import cost

# 函数1: 平移球面函数
y1 = cost(X, jj=1)

# 函数2: Schwefel问题1.2
y2 = cost(X, jj=2)

# 函数3: 带噪声的Schwefel问题1.2
y3 = cost(X, jj=3)
```

---

## 常见问题

### Q1: 如何选择合适的参数？

**基本参数建议:**
- `n_pop`: 通常30-100之间，问题越复杂可以适当增大
- `max_it`: 根据计算预算，建议至少1000次
- `var_min/var_max`: 根据实际问题的搜索范围设定

**Warm-start参数建议:**
- `alpha_mix`: 0.3-0.5较好（30-50%随机）
- `n_cand`: 10-50倍种群大小
- `sampling_method`: 'lhs'通常优于'uniform'

### Q2: Warm-start什么时候有效？

**适用场景:**
✅ 低维问题（<10维）
✅ 有相似历史优化经验
✅ 计算成本高的黑盒函数
✅ 需要快速收敛的场景

**不适用场景:**
❌ 高维简单测试函数
❌ 算法本身已经很强
❌ 有充足的迭代预算

### Q3: 如何加速训练？

```python
# 1. 减少任务数和样本数
D_meta = generate_meta_training_data(n_tasks=20, n_samples_per_task=1000)

# 2. 使用随机森林代替MLP（更快但可能精度低）
surrogate = MetaSurrogate(model_type='rf')

# 3. 减小网络规模
surrogate = MetaSurrogate(hidden_layers=(128, 64))
```

### Q4: 如何处理不同尺度的变量？

```python
# 方法1: 标准化到[-1, 1]
def standardize_vars(x_original, bounds_original):
    # bounds_original: [(min1, max1), (min2, max2), ...]
    x_std = np.zeros_like(x_original)
    for i, (vmin, vmax) in enumerate(bounds_original):
        x_std[:, i] = 2 * (x_original[:, i] - vmin) / (vmax - vmin) - 1
    return x_std

# 方法2: 在目标函数内部处理
def my_func_with_scaling(x, func_num):
    # x 是标准化的变量
    x_original = x.copy()
    x_original[:, 0] = x[:, 0] * 1000  # 第一个变量放大1000倍
    x_original[:, 1] = x[:, 1] * 0.01  # 第二个变量缩小100倍
    # ... 计算目标函数
    return result
```

---

## 完整示例

查看 `examples/` 目录下的完整示例：

- `examples/kla_warmstart_demo.py` - 完整的对比实验
- `examples/test_improved_warmstart.py` - 快速测试脚本

运行示例：
```bash
cd /path/to/KLA
python examples/test_improved_warmstart.py
```

---

## 技术支持

- 📖 详细文档: `docs/README_warmstart.md`
- 🐛 问题报告: GitHub Issues
- 📧 联系方式: Nimakhan@berkeley.edu
