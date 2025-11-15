# Kirchhoff's Law Algorithm (KLA) - Python实现

## 📋 简介

这是 **Kirchhoff's Law Algorithm (KLA)** 的 Python 实现版本。KLA 是一种新颖的受物理启发的非参数元启发式优化算法，具有warm-start初始化增强功能。

## 📁 项目结构

```
KLA/
├── src/                          # 核心源代码
│   ├── __init__.py              # 包初始化
│   ├── kla.py                   # KLA主算法
│   ├── cost.py                  # 测试函数
│   └── warmstart/               # Warm-start模块
│       ├── __init__.py
│       ├── meta_surrogate.py    # 元学习代理模型
│       └── warm_start.py        # Warm-start初始化
├── examples/                     # 示例代码
│   ├── kla_warmstart_demo.py   # 完整演示程序
│   └── test_improved_warmstart.py # 快速测试
├── docs/                         # 文档
│   ├── README_warmstart.md      # Warm-start详细文档
│   ├── warmstart_analysis.md    # 问题分析
│   ├── improvement_summary.md   # 改进总结
│   └── warm-start改进.md        # 原始设计文档
├── results/                      # 实验结果
│   ├── kla_convergence.png
│   └── warmstart_comparison_*.png
├── models/                       # 训练好的模型
│   └── surrogate_model.pkl
├── tests/                        # 测试文件（待添加）
├── requirements.txt              # 依赖包
├── license.txt                   # 许可证
└── README.md                     # 本文件
```

## 作者

- **Mojtaba Ghasemi**
- **Co-author: Nima Khodadadi** (University of California Berkeley)
- **Email:** Nimakhan@berkeley.edu
- **Homepage:** https://nimakhodadadi.com

## 引用

如果您使用此代码，请引用以下论文：

```
Ghasemi, M, Khodadadi, N. et al.
Kirchhoff's law algorithm (KLA): a novel physics-inspired 
non-parametric metaheuristic algorithm for optimization problems
Artificial Intelligence Review.
https://doi.org/10.1007/s10462-025-11289-5
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：
- numpy >= 1.21.0
- matplotlib >= 3.4.0
- scikit-learn >= 1.0.0 (用于warm-start)
- scipy >= 1.7.0 (用于warm-start)

### 2. 基本使用

#### 标准KLA算法

```python
from src import kla_optimize, cost

# 定义优化问题
n_var = 30          # 决策变量数量
var_min = -100      # 变量下界
var_max = 100       # 变量上界
max_it = 3000       # 最大迭代次数
n_pop = 50          # 种群大小
func_num = 1        # 测试函数编号 (1, 2, 或 3)

# 运行优化
best_sol, best_cost_history = kla_optimize(
    cost_function=cost,
    n_var=n_var,
    var_min=var_min,
    var_max=var_max,
    max_it=max_it,
    n_pop=n_pop,
    func_num=func_num
)

print(f"最优解: {best_sol.position}")
print(f"最优成本: {best_sol.cost}")
```

#### 使用Warm-start增强

```python
from src import kla_optimize, cost
from src.warmstart import MetaSurrogate, generate_meta_training_data

# 训练surrogate模型（一次性）
D_meta = generate_meta_training_data(n_tasks=50, n_samples_per_task=2000)
surrogate = MetaSurrogate(model_type='mlp', hidden_layers=(256, 128, 64, 32))
surrogate.train(D_meta)

# 使用warm-start运行KLA
best_sol, history = kla_optimize(
    cost_function=cost,
    n_var=30,
    var_min=-100,
    var_max=100,
    max_it=3000,
    n_pop=50,
    func_num=1,
    surrogate=surrogate,
    use_warm_start=True,
    warm_start_params={
        'n_cand': 2000,
        'alpha_mix': 0.5,
        'sampling_method': 'lhs'
    }
)
```

### 3. 运行示例

```bash
# 快速测试
python examples/test_improved_warmstart.py

# 完整演示
python examples/kla_warmstart_demo.py
```

## 📚 文档

详细文档位于 `docs/` 目录：

- **README_warmstart.md** - Warm-start功能完整指南
- **warmstart_analysis.md** - 性能分析和问题讨论
- **improvement_summary.md** - 改进措施总结

## 测试函数

项目包含 3 个标准测试函数：

1. **Basic Shifted Sphere Function** - 简单的球面函数
2. **Basic Schwefel's Problem 1.2** - Schwefel 问题 1.2
3. **Basic Schwefel's Problem 1.2 with Noise** - 带噪声的 Schwefel 问题 1.2

## 参数说明

- `n_var`: 决策变量的数量（维度）
- `var_min`: 决策变量的下界
- `var_max`: 决策变量的上界
- `max_it`: 最大函数评估次数
- `n_pop`: 种群大小（解的数量）
- `func_num`: 测试函数编号（1、2 或 3）

## 输出

运行程序后会：
1. 在控制台显示每次迭代的最优成本
2. 输出每个测试函数的统计结果（均值、最优值、标准差）
3. 生成收敛曲线图并保存为 `kla_convergence.png`

## 与 MATLAB 版本的差异

从 MATLAB 转换到 Python 时的主要变化：

1. 使用 NumPy 替代 MATLAB 的矩阵运算
2. 使用 Matplotlib 替代 MATLAB 的绘图功能
3. 使用类（Solution）来组织数据结构
4. 使用函数而非脚本的方式组织代码

## 许可证

本软件采用 BSD 许可证。详见 `license.txt` 文件。

## 转换说明

本项目由 MATLAB 代码转换而来。所有核心算法逻辑保持不变，确保与原始 MATLAB 实现的一致性。
