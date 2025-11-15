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


def cost(x, jj):
    """
    优化问题的成本函数。
    
    参数:
    -----------
    x : numpy.ndarray
        输入数组，形状为 (ps, D)，其中 ps 是种群大小，D 是维度
    jj : int
        测试函数编号 (1, 2, 或 3)
    
    返回:
    --------
    z : numpy.ndarray
        每个解的成本值
    """
    
    ps, D = x.shape
    
    # 1. 基本平移球面函数
    if jj == 1:
        x_shifted = x - ps
        z = np.sum(x_shifted**2, axis=1)
    
    # 2. 基本 Schwefel 问题 1.2
    elif jj == 2:
        x_shifted = x - ps
        z = np.zeros(ps)
        for i in range(D):
            z += np.sum(x_shifted[:, :i+1], axis=1)**2
    
    # 3. 带噪声的基本 Schwefel 问题 1.2
    elif jj == 3:
        x_shifted = x - ps
        z = np.zeros(ps)
        for i in range(D):
            z += np.sum(x_shifted[:, :i+1], axis=1)**2
        z = z * (1 + 0.4 * np.abs(np.random.randn(ps)))
    
    else:
        raise ValueError(f"无效的函数编号: {jj}。必须是 1、2 或 3。")
    
    return z
