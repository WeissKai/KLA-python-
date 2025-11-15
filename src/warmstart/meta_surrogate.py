"""
元学习Surrogate模型模块
用于训练一个可以快速预测目标函数形状的代理模型
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle


class MetaSurrogate:
    """
    元学习代理模型类
    用于跨任务学习，生成更好的初始种群
    """
    
    def __init__(self, model_type='mlp', hidden_layers=(64, 32), random_state=42):
        """
        初始化元学习代理模型
        
        参数:
        -----------
        model_type : str
            模型类型，'mlp' 或 'rf' (随机森林)
        hidden_layers : tuple
            MLP的隐藏层结构
        random_state : int
            随机种子
        """
        self.model_type = model_type
        self.random_state = random_state
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        
        if model_type == 'mlp':
            self.model = MLPRegressor(
                hidden_layer_sizes=hidden_layers,
                activation='relu',
                solver='adam',
                max_iter=1000,           # 增加迭代次数
                random_state=random_state,
                early_stopping=True,
                validation_fraction=0.1,
                learning_rate_init=0.001,
                alpha=0.0001            # L2正则化，防止过拟合
            )
        elif model_type == 'rf':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=random_state,
                n_jobs=-1
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        self.is_fitted = False
    
    def train(self, D_meta, normalize_y=True, verbose=True):
        """
        训练元学习模型
        
        参数:
        -----------
        D_meta : list of dict
            元训练数据集，每个字典包含 'X' 和 'y'
            例如: [{'X': np.array, 'y': np.array}, ...]
        normalize_y : bool
            是否对y值进行归一化
        verbose : bool
            是否显示训练信息
        """
        # 合并所有任务的数据
        X_all = []
        y_all = []
        
        for task_data in D_meta:
            X_task = task_data['X']
            y_task = task_data['y']
            
            # 可选：对每个任务的y做归一化，模拟不同尺度的函数
            if normalize_y and len(y_task) > 1:
                y_mean = np.mean(y_task)
                y_std = np.std(y_task)
                if y_std > 1e-10:
                    y_task = (y_task - y_mean) / y_std
            
            X_all.append(X_task)
            y_all.append(y_task)
        
        X_all = np.vstack(X_all)
        y_all = np.concatenate(y_all)
        
        if verbose:
            print(f"元学习训练数据: {len(X_all)} 个样本, 维度: {X_all.shape[1]}")
        
        # 标准化输入
        X_all = self.scaler_x.fit_transform(X_all)
        
        # 训练模型
        self.model.fit(X_all, y_all)
        self.is_fitted = True
        
        if verbose:
            # 计算训练集上的R²分数
            score = self.model.score(X_all, y_all)
            print(f"元模型训练完成，R² 分数: {score:.4f}")
    
    def predict(self, X):
        """
        预测候选点的质量（代理目标函数值）
        
        参数:
        -----------
        X : numpy.ndarray
            候选点，形状 (n_samples, n_features)
        
        返回:
        --------
        y_pred : numpy.ndarray
            预测的目标函数值
        """
        if not self.is_fitted:
            raise RuntimeError("模型尚未训练，请先调用 train() 方法")
        
        X_scaled = self.scaler_x.transform(X)
        return self.model.predict(X_scaled)
    
    def save(self, filepath):
        """保存模型到文件"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler_x': self.scaler_x,
                'scaler_y': self.scaler_y,
                'model_type': self.model_type,
                'is_fitted': self.is_fitted
            }, f)
        print(f"模型已保存到: {filepath}")
    
    def load(self, filepath):
        """从文件加载模型"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.scaler_x = data['scaler_x']
        self.scaler_y = data['scaler_y']
        self.model_type = data['model_type']
        self.is_fitted = data['is_fitted']
        print(f"模型已从 {filepath} 加载")


def generate_meta_training_data(n_tasks=10, n_samples_per_task=100, n_var=30, 
                                var_min=-100, var_max=100):
    """
    生成元训练数据集（使用多个测试函数）
    
    参数:
    -----------
    n_tasks : int
        任务数量
    n_samples_per_task : int
        每个任务的样本数
    n_var : int
        变量维度
    var_min : float
        变量下界
    var_max : float
        变量上界
    
    返回:
    --------
    D_meta : list of dict
        元训练数据集
    """
    from ..cost import cost
    
    D_meta = []
    
    # 使用现有的3个测试函数，并生成一些变体
    base_functions = [1, 2, 3]
    
    for task_id in range(n_tasks):
        # 随机选择一个基础函数
        func_num = base_functions[task_id % len(base_functions)]
        
        # 改进的采样策略：混合随机采样和有偏采样
        n_random = int(n_samples_per_task * 0.7)  # 70%随机
        n_biased = n_samples_per_task - n_random  # 30%偏向好的区域
        
        # 随机采样
        X_random = np.random.uniform(var_min, var_max, (n_random, n_var))
        
        # 偏向采样：在较好区域（接近原点或某些特定点）附近采样
        center = np.random.uniform(-10, 10, (1, n_var))  # 随机中心点
        std = np.random.uniform(20, 50)  # 随机标准差
        X_biased = center + np.random.randn(n_biased, n_var) * std
        X_biased = np.clip(X_biased, var_min, var_max)
        
        # 合并样本
        X_task = np.vstack([X_random, X_biased])
        
        # 计算目标函数值
        y_task = cost(X_task, func_num)
        
        # 添加随机平移和缩放，增加任务多样性
        if task_id >= len(base_functions):
            scale = np.random.uniform(0.5, 2.0)
            shift = np.random.uniform(-100, 100)
            y_task = y_task * scale + shift
        
        D_meta.append({'X': X_task, 'y': y_task})
    
    return D_meta
