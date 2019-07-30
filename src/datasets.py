import numpy as np

def bc_nonlinear(n=100, width=1.5):
    """
    Toy_Dataset_I
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
    x = np.random.uniform(-width, width, (n,2))
    noize = np.random.normal(0, 0.1, n)
    r = np.sum(x*x, axis=1) + noize
    y = np.where(abs(r - width) < width/2, 1, -1)

    return x, y

def bc_linear(n=100):
    """
    Toy_Dataset_II
    二値判別のテストデータセット（線形）
    
    Parameters
    ------------------------------
    n: Int = サンプル数

    Returns
    ------------------------------
    x: np.array(n, 2) = 各座標が標準正規分布のデータ群
    y: np.array(n) = {+1, -1}^n
    """
    dim=2
    a = np.random.normal(size = (dim,))
    noize = np.random.normal(0, 0.8, n)
    x = np.random.normal(size = (n, dim))
    l = x@a + noize
    y = np.where(l > 0, 1, -1)
    return x, y

def bc_linear_4(n=100):
    """
    Toy_Dataset_IV
    二値判別のテストデータセット（線形）
    
    Parameters
    ------------------------------
    n: Int = サンプル数

    Returns
    ------------------------------
    x: np.array(n, 4) = データ群
    y: np.array(n) = {+1, -1}^n
    """
    noize = 0.5 * np.random.randn(n)
    x = 3 * (np.random.rand(n, 4) - 0.5)
    l = x@[0, 2, -1, 0] + 0.5 + noize
    y = np.where(l > 0, 1, -1)

    return x, y

def mc_linear(n=100):
    """
    Toy_Dataset_V
    多値判別のテストデータセット（線形）
    
    Parameters
    ------------------------------
    n: Int = サンプル数

    Returns
    ------------------------------
    x: np.array(n, 4)
        データ群
    y: np.array(n) = {0, 1, 2}^n
        ラベル(Xの第2,3成分についてほぼ線形分離可能)
    """
    X = 3 * (np.random.rand(n, 4) - 0.5)
    W = np.array([
        [ 2, -1, 0.5],
        [-3,  2,   1],
        [ 1,  2,   3]])
    noize = 0.5 * np.random.randn(n, 3)
    y = np.argmax(np.insert(X[:, 1:3], 2, 1, axis=1)@W + noize, axis=1)
    return X, y
