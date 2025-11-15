# Kirchhoff's Law Algorithm (KLA) - Python Implementation

## 简介

这是 **Kirchhoff's Law Algorithm (KLA)** 的 Python 实现版本。KLA 是一种新颖的受物理启发的非参数元启发式优化算法。

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

## 安装

### 1. 安装依赖包

```bash
pip install -r requirements.txt
```

或者手动安装：

```bash
pip install numpy matplotlib
```

## 使用方法

### 基本使用

直接运行主程序：

```bash
python kla.py
```

这将在 3 个测试函数上运行 KLA 算法，每个函数独立执行 2 次。

### 自定义使用

```python
from kla import kla_optimize
from cost import cost

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

## 文件说明

- **kla.py** - 主程序文件，包含 KLA 算法的实现
- **cost.py** - 测试函数定义
- **requirements.txt** - Python 依赖包列表
- **license.txt** - 许可证文件

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
