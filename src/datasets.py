import numpy as np

def bc_nonlinear(n=100, width=1.5):
    """
    二値判別のテストデータセット（非線形）
    
    Parameters
    ------------------------------
    n: Int
        サンプル数
    width: float
        乱数分布の幅

    Returns
    ------------------------------
    x: np.array(n, 2)
        2次元平面内のn個の点の座標
    y: np.array(n)
        ラベル {+1, -1}
    """
    x = np.random.uniform(-width, width, (n, 2))
    noize = np.random.normal(0, 0.1, n)
    r = np.sum(x*x, axis=1) + noize
    y = np.where(abs(r - width) < width/2, 1, -1)

    return x, y

def bc_linear(n=100,dim=2):
    """
    二値判別のテストデータセット（線形）
    
    Parameters
    ------------------------------
    n: Int = サンプル数
    dim: Int = データの次元

    Returns
    ------------------------------
    x: np.array(n, dim) = 各座標が標準正規分布のデータ群
    y: np.array(n) = {+1, -1}^n
    """
    a = np.random.normal(size = (dim,))
    noize = np.random.normal(0, 0.8, n)
    x = np.random.normal(size = (n, dim))
    l = x@a + noize
    y = np.where(l > 0, 1, -1)
    return x, y


