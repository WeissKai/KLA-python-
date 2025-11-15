"""
___________________________________________________________________
                 Kirchhoff's law algorithm (KLA)                      
                                                                       
                                                                       
                 Developed in Python                                   
                                                                       
                     Author and programmer                            
                                                                       
               ---------------------------------                      
                         Mojtaba Ghasemi                              
    Co:   Nima Khodadadi (ʘ‿ʘ) University of California Berkeley      
                            e-Mail                                    
               ---------------------------------                      
                     Nimakhan@berkeley.edu                            
                                                                       
                                                                       
                           Homepage                                   
               ---------------------------------                      
                   https://nimakhodadadi.com                          
                                                                       
                                                                       
                                                                       
                                                                       
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                          Citation                                    
Ghasemi, M, Khodadadi, N. et al.                                      
Kirchhoff's law algorithm (KLA): a novel physics-inspired             
non-parametric metaheuristic algorithm for optimization problems      
Artificial Intelligence Review.                                       
https://doi.org/10.1007/s10462-025-11289-5                            
---------------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
from cost import cost

# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class Solution:
    """表示优化问题中的一个解的类。"""
    def __init__(self):
        self.position = None
        self.cost = np.inf


def kla_optimize(cost_function, n_var, var_min, var_max, max_it=3000, n_pop=50, func_num=1):
    """
    基尔霍夫定律算法 (KLA) 优化器。
    
    参数:
    -----------
    cost_function : callable
        要最小化的目标函数
    n_var : int
        决策变量的数量
    var_min : float
        决策变量的下界
    var_max : float
        决策变量的上界
    max_it : int
        最大函数评估次数 (默认: 3000)
    n_pop : int
        种群大小 (默认: 50)
    func_num : int
        测试函数编号 (默认: 1)
    
    返回:
    --------
    best_sol : Solution
        找到的最优解
    best_cost_history : numpy.ndarray
        最优成本的历史记录
    """
    
    # 初始化参数
    ebs = np.finfo(float).eps  # 机器精度 (相当于 MATLAB 的 realmin)
    
    # 初始化种群
    pop = [Solution() for _ in range(n_pop)]
    
    # 初始化最优解
    best_sol = Solution()
    best_sol.cost = np.inf
    TT = -np.inf
    
    # 创建初始种群
    best_cost_history = []
    
    for i in range(n_pop):
        pop[i].position = np.random.uniform(var_min, var_max, (1, n_var))
        pop[i].cost = cost_function(pop[i].position, func_num)[0]
        
        if pop[i].cost <= best_sol.cost:
            best_sol.position = pop[i].position.copy()
            best_sol.cost = pop[i].cost
        
        if pop[i].cost > TT:
            TT = pop[i].cost
        
        best_cost_history.append(best_sol.cost)
    
    # KLA 主循环
    it = n_pop
    
    while it <= max_it:
        for i in range(n_pop):
            # 选择三个不同于 i 的随机个体
            A = np.random.permutation(n_pop)
            A = A[A != i]
            
            a = A[0]
            b = A[1]
            jj = A[2]
            
            # 计算电荷相关参数
            q = ((pop[i].cost - pop[jj].cost) + ebs) / (abs(pop[i].cost - pop[jj].cost) + ebs)
            Q = (pop[i].cost - pop[a].cost) / (abs(pop[i].cost - pop[a].cost) + ebs)
            Q2 = (pop[i].cost - pop[b].cost) / (abs(pop[i].cost - pop[b].cost) + ebs)
            
            q1 = ((pop[jj].cost) / (pop[i].cost + ebs)) ** (2 * np.random.rand())
            Q1 = ((pop[a].cost) / (pop[i].cost + ebs)) ** (2 * np.random.rand())
            Q21 = ((pop[b].cost) / (pop[i].cost + ebs)) ** (2 * np.random.rand())
            
            # 计算搜索方向
            S1 = q1 * q * np.random.rand(1, n_var) * (pop[jj].position - pop[i].position)
            S2 = Q * Q1 * np.random.rand(1, n_var) * (pop[a].position - pop[i].position)
            S3 = Q2 * Q21 * np.random.rand(1, n_var) * (pop[b].position - pop[i].position)
            S = (np.random.rand() + np.random.rand()) * S1 + \
                (np.random.rand() + np.random.rand()) * S2 + \
                (np.random.rand() + np.random.rand()) * S3
            
            # 更新位置
            new_position = pop[i].position + S
            
            # 应用边界约束
            new_position = np.maximum(new_position, var_min)
            new_position = np.minimum(new_position, var_max)
            
            # 评估新解
            new_cost = cost_function(new_position, func_num)[0]
            
            # 如果更好则更新
            if new_cost <= pop[i].cost:
                pop[i].position = new_position.copy()
                pop[i].cost = new_cost
                
                if pop[i].cost <= best_sol.cost:
                    best_sol.position = pop[i].position.copy()
                    best_sol.cost = pop[i].cost
            
            it += 1
            best_cost_history.append(best_sol.cost)
            
            if it > max_it:
                break
        
        # 显示迭代信息
        print(f'迭代 {it}: 最优成本 = {best_sol.cost}')
    
    return best_sol, np.array(best_cost_history)


def run_kla_experiments(n_functions=3, n_runs=2):
    """
    在多个测试函数上运行 KLA 实验。
    
    参数:
    -----------
    n_functions : int
        要运行的测试函数数量 (默认: 3)
    n_runs : int
        每个函数的独立运行次数 (默认: 2)
    """
    
    # 问题定义
    n_var = 30  # 决策变量数量
    var_min = -100  # 下界
    var_max = 100  # 上界
    max_it = 3000  # 最大函数评估次数
    n_pop = 50  # 种群大小
    
    # 存储结果
    mean_results = []
    best_results = []
    std_results = []
    
    plt.figure(figsize=(10, 6))
    
    for nf in range(1, n_functions + 1):
        print(f"\n{'='*60}")
        print(f"运行测试函数 {nf}")
        print(f"{'='*60}")
        
        cost_results = []
        all_histories = []
        
        for run in range(n_runs):
            print(f"\n第 {run + 1}/{n_runs} 次运行")
            print(f"-" * 60)
            
            # 运行 KLA 优化
            best_sol, best_cost_history = kla_optimize(
                cost_function=cost,
                n_var=n_var,
                var_min=var_min,
                var_max=var_max,
                max_it=max_it,
                n_pop=n_pop,
                func_num=nf
            )
            
            cost_results.append(best_sol.cost)
            all_histories.append(best_cost_history)
        
        # 计算统计数据
        mean_cost = np.mean(cost_results)
        best_cost = np.min(cost_results)
        std_cost = np.std(cost_results)
        
        mean_results.append(mean_cost)
        best_results.append(best_cost)
        std_results.append(std_cost)
        
        # 绘制收敛曲线
        all_histories_array = np.array(all_histories)
        mean_history = np.mean(all_histories_array, axis=0)
        plt.plot(np.log(mean_history), linewidth=2, label=f'函数 {nf}')
        
        # 打印结果
        print(f"\n{'='*60}")
        print(f"函数 {nf} 结果:")
        print(f"均值: {mean_cost}")
        print(f"最优: {best_cost}")
        print(f"标准差: {std_cost}")
        print(f"{'='*60}")
    
    # 完成绘图
    plt.xlabel('迭代次数')
    plt.ylabel('对数(最优成本)')
    plt.title('KLA 收敛曲线')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('kla_convergence.png', dpi=300)
    plt.show()
    
    return mean_results, best_results, std_results


if __name__ == "__main__":
    # 运行实验
    mean_results, best_results, std_results = run_kla_experiments(n_functions=3, n_runs=2)
    
    print("\n" + "="*60)
    print("最终总结")
    print("="*60)
    for i in range(len(mean_results)):
        print(f"函数 {i+1}:")
        print(f"  均值: {mean_results[i]:.6e}")
        print(f"  最优: {best_results[i]:.6e}")
        print(f"  标准差:  {std_results[i]:.6e}")
        print("-"*60)
