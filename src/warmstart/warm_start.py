"""
Warm-start初始化模块
使用元学习的surrogate模型生成高质量的初始种群
"""

import numpy as np
from scipy.spatial.distance import cdist


def warm_start_initialization(surrogate, search_space, n_pop, n_cand=None, 
                              alpha_mix=0.3, diversity_threshold=None, 
                              sampling_method='uniform', verbose=False):
    """
    使用surrogate模型生成warm-start初始种群
    
    参数:
    -----------
    surrogate : MetaSurrogate
        训练好的元学习代理模型
    search_space : tuple
        搜索空间 (var_min, var_max, n_var)
    n_pop : int
        种群大小
    n_cand : int
        候选点数量，默认为 20 * n_pop
    alpha_mix : float
        混合比例，保留的随机初始化比例（0到1之间）
        例如 0.3 表示 70% 用 surrogate 选择，30% 随机
    diversity_threshold : float
        多样性阈值，用于筛选候选点，默认为搜索空间对角线长度的 5%
    sampling_method : str
        候选点采样方法: 'uniform', 'lhs', 'sobol'
    verbose : bool
        是否显示详细信息
    
    返回:
    --------
    X_init : numpy.ndarray
        初始种群，形状 (n_pop, n_var)
    """
    var_min, var_max, n_var = search_space
    
    if n_cand is None:
        n_cand = 20 * n_pop
    
    if verbose:
        print(f"生成 {n_cand} 个候选点...")
    
    # 1. 生成大量候选点
    candidates = generate_candidates(var_min, var_max, n_var, n_cand, sampling_method)
    
    # 2. 使用surrogate预测每个候选点的质量
    if verbose:
        print("使用surrogate模型评估候选点...")
    
    y_pred = surrogate.predict(candidates)
    
    # 3. 按预测值排序（假设最小化问题）
    sorted_indices = np.argsort(y_pred)
    candidates_sorted = candidates[sorted_indices]
    
    # 4. 计算多样性阈值（增加阈值以获得更好的多样性）
    if diversity_threshold is None:
        # 使用搜索空间对角线长度的10%（增加以提高多样性）
        diagonal = np.sqrt(n_var * (var_max - var_min) ** 2)
        diversity_threshold = 0.10 * diagonal
    
    if verbose:
        print(f"多样性阈值: {diversity_threshold:.4f}")
    
    # 5. 从前面的候选点中选择，同时保证多样性
    n_surrogate = int((1 - alpha_mix) * n_pop)
    n_random = n_pop - n_surrogate
    
    # 增加考虑的候选点范围，避免只看最好的几个
    selected = diversity_selection(candidates_sorted, n_surrogate, diversity_threshold, 
                                   max_candidates=min(10 * n_pop, n_cand))
    
    if verbose:
        print(f"从surrogate中选择了 {len(selected)} 个点")
    
    # 6. 补充随机点以保持探索能力
    if n_random > 0:
        random_points = np.random.uniform(var_min, var_max, (n_random, n_var))
        if verbose:
            print(f"添加 {n_random} 个随机点")
    else:
        random_points = np.array([]).reshape(0, n_var)
    
    # 7. 合并得到最终初始种群
    if len(selected) > 0 and len(random_points) > 0:
        X_init = np.vstack([selected, random_points])
    elif len(selected) > 0:
        X_init = selected
    else:
        X_init = random_points
    
    # 确保正好是 n_pop 个点
    if len(X_init) < n_pop:
        # 如果不够，补充随机点
        n_extra = n_pop - len(X_init)
        extra_points = np.random.uniform(var_min, var_max, (n_extra, n_var))
        X_init = np.vstack([X_init, extra_points])
    elif len(X_init) > n_pop:
        # 如果太多，截取前 n_pop 个
        X_init = X_init[:n_pop]
    
    if verbose:
        print(f"生成的初始种群大小: {X_init.shape}")
    
    return X_init


def generate_candidates(var_min, var_max, n_var, n_cand, method='uniform'):
    """
    生成候选点
    
    参数:
    -----------
    var_min : float
        变量下界
    var_max : float
        变量上界
    n_var : int
        变量维度
    n_cand : int
        候选点数量
    method : str
        采样方法: 'uniform', 'lhs', 'sobol'
    
    返回:
    --------
    candidates : numpy.ndarray
        候选点数组
    """
    if method == 'uniform':
        candidates = np.random.uniform(var_min, var_max, (n_cand, n_var))
    
    elif method == 'lhs':
        # 拉丁超立方采样
        try:
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=n_var)
            samples = sampler.random(n=n_cand)
            candidates = var_min + (var_max - var_min) * samples
        except ImportError:
            print("警告: scipy.stats.qmc 不可用，使用均匀采样")
            candidates = np.random.uniform(var_min, var_max, (n_cand, n_var))
    
    elif method == 'sobol':
        # Sobol序列采样
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=n_var, scramble=True)
            samples = sampler.random(n=n_cand)
            candidates = var_min + (var_max - var_min) * samples
        except ImportError:
            print("警告: scipy.stats.qmc 不可用，使用均匀采样")
            candidates = np.random.uniform(var_min, var_max, (n_cand, n_var))
    
    else:
        raise ValueError(f"不支持的采样方法: {method}")
    
    return candidates


def diversity_selection(candidates, n_select, diversity_threshold, max_candidates=None):
    """
    从候选点中选择保证多样性的子集
    
    参数:
    -----------
    candidates : numpy.ndarray
        候选点数组（已按质量排序）
    n_select : int
        需要选择的点数
    diversity_threshold : float
        多样性阈值
    max_candidates : int
        最多考虑前多少个候选点
    
    返回:
    --------
    selected : numpy.ndarray
        选择的点
    """
    if max_candidates is not None:
        candidates = candidates[:max_candidates]
    
    selected = []
    
    for i, candidate in enumerate(candidates):
        if len(selected) == 0:
            # 第一个点直接选择
            selected.append(candidate)
        else:
            # 计算与已选点的最小距离
            distances = cdist([candidate], selected, metric='euclidean')[0]
            min_distance = np.min(distances)
            
            if min_distance > diversity_threshold:
                selected.append(candidate)
            
            if len(selected) >= n_select:
                break
    
    if len(selected) > 0:
        return np.array(selected)
    else:
        return np.array([]).reshape(0, candidates.shape[1])
