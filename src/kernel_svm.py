import numpy as np
import matplotlib.pyplot as plt
import seaborn; seaborn.set()
from datasets import *
from draw_fig import *

class kernel_SVM:
    """
        Parameters
        ---------------
        X: np.array(n,d)
            データ
        y: np.array(n)
            ラベル
        c: float
            正則化項の係数
        sigma: float
            gaussian kernelの

        How to fit
        ----------------
        project_gradient_descentを実行する
    """
    def __init__(self, X, y, c, sigma=1):
        self.X = X
        self.y = y
        self.c = c
        self.gram = self.gaussian_kernel(sigma)

    def dual_Lagrange_function(self, a):
        n = len(self.y)
        K = self.y * self.gram * self.y.reshape(n,1)
        return - 1/2 * a@K@a + np.sum(a)

    def grad_dLf(self, a):
        n = len(self.y)
        K = self.y * self.gram * self.y.reshape(n,1)
        return - K@a + 1
            
    def gk(self, x, sigma):
        "core part of gaussian kernel"
        n, d = self.X.shape
        pivot = x
        XX = np.sum(self.X*self.X, axis=1).reshape(n,1)
        Xpiv = self.X@pivot.T
        pivpiv = np.sum(pivot*pivot, axis=1)
        return np.exp(- (XX - 2 * Xpiv + pivpiv) / (2 * sigma ** 2))

    def gaussian_kernel(self, sigma):
        """
        Parameter
        ----------
        sigma: float
            ハイパーパラメータ
        
        Returns
        ----------
        gram_matrix: np.array(size = (n,n))
            Gaussian kernelによるGram行列
        """
        return self.gk(self.X, sigma)

    def projected_gradient_descent(self, a0=None, learning_rate=0.01, eps=1e-6, max_itr=10000):
        """
        射影勾配法による双対問題の最適解の求解関数

        Parameters
        ---------------
        X: np.array(n,d)
            データ
        y: np.array(n)
            ラベル
        c: float
            正則化項の係数
        a0: np.array(n)
            初期値（与えられない場合は乱数）
        learning_rate: float
            学習係数
        max_itr: Int
            最大反復回数
        eps: float
            反復終了条件

        Returns
        ------------------
        alpha: np.array(d)
            最適判別パラメータ
        alpha_log: List[np.array(d)]
            各反復でのalphaの値
        """
        n, d = self.X.shape
        alpha = np.clip(a0, 0, self.c/n) if a0 is not None else np.clip(np.random.normal(size=(n,)), 0, self.c/n)
        prev = self.dual_Lagrange_function(alpha)
        for itr in range(max_itr):
            alpha = alpha + (itr+1)**(-1) * self.grad_dLf(alpha)
            alpha = np.clip(alpha, 0, self.c/n)
            now = self.dual_Lagrange_function(alpha)
            print(f"iteration {itr}: {prev} -> {now}")
            if(abs(prev - now) < eps):
                print(f"iteration stop at {itr}")
                break
            else:
                prev = now
        return alpha

def main():
    n = 100
    c = n
    X, y = bc_linear(n)
    sigma = 0.7
    svm = kernel_SVM(X, y, c, sigma)
    a = svm.projected_gradient_descent(max_itr=10000, learning_rate=1e-7)
    ker = lambda x: svm.gk(x, sigma)
    w = y * a
    bc_plot_kernel(X, y, w, ker)


main()