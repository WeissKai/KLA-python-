"""
KLA Warm-start改进版本的示例和测试代码
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.warmstart import MetaSurrogate, generate_meta_training_data
from src import kla_optimize, cost

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong']
plt.rcParams['axes.unicode_minus'] = False


def train_surrogate_model(save_path='surrogate_model.pkl'):
    """
    训练元学习surrogate模型
    """
    print("="*60)
    print("步骤1: 训练元学习Surrogate模型")
    print("="*60)
    
    # 生成元训练数据（大幅增加数据量）
    print("\n生成元训练数据...")
    D_meta = generate_meta_training_data(
        n_tasks=50,               # 增加任务数
        n_samples_per_task=2000,  # 增加每个任务的样本数
        n_var=30,
        var_min=-100,
        var_max=100
    )
    
    print(f"生成了 {len(D_meta)} 个任务的训练数据")
    
    # 创建并训练模型（使用更大的网络）
    print("\n训练Surrogate模型...")
    surrogate = MetaSurrogate(model_type='mlp', hidden_layers=(256, 128, 64, 32))
    surrogate.train(D_meta, normalize_y=True, verbose=True)
    
    # 保存模型
    surrogate.save(save_path)
    
    return surrogate


def compare_with_and_without_warmstart(surrogate, func_num=1, n_runs=5):
    """
    比较使用和不使用warm-start的KLA性能
    """
    print("\n" + "="*60)
    print(f"步骤2: 比较测试函数 {func_num} 的性能")
    print("="*60)
    
    # 参数设置
    n_var = 30
    var_min = -100
    var_max = 100
    max_it = 3000
    n_pop = 50
    
    results_without_ws = []
    results_with_ws = []
    histories_without_ws = []
    histories_with_ws = []
    
    for run in range(n_runs):
        print(f"\n--- 第 {run+1}/{n_runs} 次运行 ---")
        
        # 不使用warm-start
        print("运行标准KLA...")
        np.random.seed(run)  # 设置种子以便复现
        best_sol, history = kla_optimize(
            cost_function=cost,
            n_var=n_var,
            var_min=var_min,
            var_max=var_max,
            max_it=max_it,
            n_pop=n_pop,
            func_num=func_num,
            use_warm_start=False
        )
        results_without_ws.append(best_sol.cost)
        histories_without_ws.append(history)
        print(f"标准KLA最终成本: {best_sol.cost:.6e}")
        
        # 使用warm-start
        print("运行Warm-start KLA...")
        np.random.seed(run)  # 使用相同种子
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
                'n_cand': 2000,              # 增加候选点数量
                'alpha_mix': 0.5,            # 降低surrogate权重：50%随机，50%surrogate
                'sampling_method': 'lhs',    # 使用拉丁超立方采样
                'verbose': False
            }
        )
        results_with_ws.append(best_sol_ws.cost)
        histories_with_ws.append(history_ws)
        print(f"Warm-start KLA最终成本: {best_sol_ws.cost:.6e}")
        print(f"改进: {(best_sol.cost - best_sol_ws.cost) / best_sol.cost * 100:.2f}%")
    
    # 统计结果
    print("\n" + "="*60)
    print("统计结果")
    print("="*60)
    print(f"标准KLA:")
    print(f"  均值: {np.mean(results_without_ws):.6e}")
    print(f"  最优: {np.min(results_without_ws):.6e}")
    print(f"  标准差: {np.std(results_without_ws):.6e}")
    
    print(f"\nWarm-start KLA:")
    print(f"  均值: {np.mean(results_with_ws):.6e}")
    print(f"  最优: {np.min(results_with_ws):.6e}")
    print(f"  标准差: {np.std(results_with_ws):.6e}")
    
    improvement = (np.mean(results_without_ws) - np.mean(results_with_ws)) / np.mean(results_without_ws) * 100
    print(f"\n平均改进: {improvement:.2f}%")
    
    # 绘制对比图
    plot_comparison(histories_without_ws, histories_with_ws, func_num)
    
    return results_without_ws, results_with_ws


def plot_comparison(histories_without_ws, histories_with_ws, func_num):
    """
    绘制对比收敛曲线
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 转换为数组
    histories_without_ws = np.array(histories_without_ws)
    histories_with_ws = np.array(histories_with_ws)
    
    # 计算平均值
    mean_without_ws = np.mean(histories_without_ws, axis=0)
    mean_with_ws = np.mean(histories_with_ws, axis=0)
    
    # 左图: 对数尺度
    ax1 = axes[0]
    ax1.plot(np.log10(mean_without_ws + 1e-10), 'b-', linewidth=2, label='标准KLA')
    ax1.plot(np.log10(mean_with_ws + 1e-10), 'r-', linewidth=2, label='Warm-start KLA')
    ax1.set_xlabel('迭代次数')
    ax1.set_ylabel('Log₁₀(最优成本)')
    ax1.set_title(f'函数 {func_num} - 对数收敛曲线')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 右图: 线性尺度（前1000次迭代）
    ax2 = axes[1]
    max_iter = min(1000, len(mean_without_ws))
    ax2.plot(mean_without_ws[:max_iter], 'b-', linewidth=2, label='标准KLA')
    ax2.plot(mean_with_ws[:max_iter], 'r-', linewidth=2, label='Warm-start KLA')
    ax2.set_xlabel('迭代次数')
    ax2.set_ylabel('最优成本')
    ax2.set_title(f'函数 {func_num} - 前期收敛对比')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'warmstart_comparison_func{func_num}.png', dpi=300)
    plt.show()
    
    print(f"\n对比图已保存: warmstart_comparison_func{func_num}.png")


def main():
    """
    主函数：完整的warm-start改进流程
    """
    print("KLA Warm-start改进版本")
    print("="*60)
    
    # 1. 训练surrogate模型（如果已有模型可以跳过）
    try:
        surrogate = MetaSurrogate()
        surrogate.load('surrogate_model.pkl')
        print("已加载预训练的Surrogate模型")
    except:
        print("未找到预训练模型，开始训练...")
        surrogate = train_surrogate_model()
    
    # 2. 在各个测试函数上进行对比实验
    all_results = {}
    
    for func_num in [1, 2, 3]:
        results_std, results_ws = compare_with_and_without_warmstart(
            surrogate, func_num=func_num, n_runs=3
        )
        all_results[func_num] = {
            'standard': results_std,
            'warmstart': results_ws
        }
    
    # 3. 总结
    print("\n" + "="*60)
    print("所有测试函数总结")
    print("="*60)
    
    for func_num in [1, 2, 3]:
        std_mean = np.mean(all_results[func_num]['standard'])
        ws_mean = np.mean(all_results[func_num]['warmstart'])
        improvement = (std_mean - ws_mean) / std_mean * 100
        
        print(f"\n函数 {func_num}:")
        print(f"  标准KLA均值: {std_mean:.6e}")
        print(f"  Warm-start均值: {ws_mean:.6e}")
        print(f"  改进: {improvement:.2f}%")


if __name__ == "__main__":
    main()
