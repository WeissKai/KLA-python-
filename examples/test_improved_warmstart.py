"""
快速测试改进后的warm-start效果
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.warmstart import MetaSurrogate, generate_meta_training_data
from src import kla_optimize, cost

print("="*60)
print("快速测试改进后的Warm-start")
print("="*60)

# 训练改进的surrogate模型
print("\n1. 训练改进的Surrogate模型...")
D_meta = generate_meta_training_data(
    n_tasks=50,
    n_samples_per_task=2000,
    n_var=30,
    var_min=-100,
    var_max=100
)
print(f"训练数据: {len(D_meta)} 个任务, 总计 {50*2000} 个样本")

surrogate = MetaSurrogate(model_type='mlp', hidden_layers=(256, 128, 64, 32))
surrogate.train(D_meta, normalize_y=True, verbose=True)

# 在函数1上快速测试
print("\n2. 在函数1上测试...")
func_num = 1
n_var = 30
var_min = -100
var_max = 100
max_it = 1500  # 减少迭代次数以加快测试
n_pop = 50

# 标准KLA
print("\n运行标准KLA...")
np.random.seed(42)
best_sol_std, history_std = kla_optimize(
    cost_function=cost,
    n_var=n_var,
    var_min=var_min,
    var_max=var_max,
    max_it=max_it,
    n_pop=n_pop,
    func_num=func_num,
    use_warm_start=False
)
print(f"标准KLA最终成本: {best_sol_std.cost:.6e}")

# Warm-start KLA
print("\n运行改进的Warm-start KLA...")
np.random.seed(42)
best_sol_ws, history_ws = kla_optimize(
    cost_function=cost,
    n_var=n_var,
    var_min=var_min,
    var_max=var_max,
    max_it=max_it,
    n_pop=n_pop,
    func_num=func_num,
    surrogate=surrogate,
    use_warm_start=True,
    warm_start_params={
        'n_cand': 2000,
        'alpha_mix': 0.5,
        'sampling_method': 'lhs',
        'verbose': True
    }
)
print(f"Warm-start KLA最终成本: {best_sol_ws.cost:.6e}")

# 结果对比
improvement = (best_sol_std.cost - best_sol_ws.cost) / best_sol_std.cost * 100
print("\n" + "="*60)
print("结果对比")
print("="*60)
print(f"标准KLA: {best_sol_std.cost:.6e}")
print(f"Warm-start KLA: {best_sol_ws.cost:.6e}")
print(f"改进: {improvement:.2f}%")

if improvement > 0:
    print("\n✅ Warm-start效果改善！")
else:
    print("\n⚠️ Warm-start仍需继续优化")
