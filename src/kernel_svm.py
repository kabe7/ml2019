import numpy as np
import matplotlib.pyplot as plt
import seaborn; seaborn.set()
from datasets import *
from draw_fig import bc_plot_kernel

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
        kernels: 'gauss', 'poly'
            カーネル関数
        param: float
            kernel関数のハイパーパラメータ

        How to fit
        ----------------
        project_gradient_descentを実行する
    """

    def __init__(self, X, y, lam, method_name, param):
        kernels = {
        'gauss': self.gaussian_kernel,
        'poly': self.polynomial_kernel
        }

        self.X = X
        self.y = y
        self.lam = lam
        self.gram = kernels[method_name](param)

    def primal_function(self, w):
        "主問題の目的関数"
        loss = np.clip(1 - self.y * (self.gram@w), 0, None)
        return np.sum(loss) + self.lam * w@w

    def dual_Lagrange_function(self, a):
        "双対問題の目的関数"
        n = len(self.y)
        K = self.y * self.gram * self.y.reshape(n,1)
        return - 1/(4 * self.lam) * a@K@a + np.sum(a)

    def grad_dLf(self, a):
        n = len(self.y)
        K = self.y * self.gram * self.y.reshape(n,1)
        return - 1/(2 * self.lam) * K@a + 1

    def dual2primal(self, a):
        "双対問題の解を主問題の解に変換"
        return self.y * a

    def gk(self, x, sigma):
        """
        Parameter:
        ----------
        x: np.array(*,d)
            入力データ
        sigma: float
            ハイパーパラメータ

        Returns:
        -----------
        x_K: np.array(n, *)
            {x_K}_i = exp(- ||x - x_i||^2 / 2σ^2 )
        """
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

    def poly(self, x, degree):
        """
        core part of polynomial kernel
        
        Parameter:
        -------------
        x: np.array(*,d)
            入力データ
        degree: int(natural number)
            次数
        """
        return (self.X@x.T + 1)**degree

    def polynomial_kernel(self, degree):
        """
        Parameter
        ----------
        degree: positive int
            ハイパーパラメータ
        
        Returns
        ----------
        gram_matrix: np.array(size = (n,n))
            Polynomial kernelによるGram行列
        """
        return self.poly(self.X, degree)

    def projected_gradient_descent(self, a0=None, learning_rate=0.01, eps=1e-6, max_itr=10000):
        """
        射影勾配法による双対問題の最適解の求解関数

        Parameters
        ---------------
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
        a: np.array(n)
            最適判別パラメータ
        """
        if not callable(learning_rate):
            eta = lambda itr: learning_rate
        else:
            eta = learning_rate

        n, _ = self.X.shape
        a = a0 if a0 is not None else np.random.normal(size=(n,))
        a = np.clip(a, 0, 1)
        prev = self.dual_Lagrange_function(a)
        for itr in range(max_itr):
            a += eta(itr) * self.grad_dLf(a)
            a = np.clip(a, 0, 1)
            now = self.dual_Lagrange_function(a)
            print(f"iteration {itr}: {prev} -> {now}", end='\r')
            if(abs(prev - now) < eps):
                print(f"\niteration stop at {itr}")
                break
            else:
                prev = now
        return a

def main():
    n = 100 # number of data
    X, y = bc_nonlinear(n) # dataset
    lam, sigma, degree = 1, 0.6, 3 # hyper-parameters

    # build kernel_SVM instance
    svm = kernel_SVM(X, y, lam, 'gauss', sigma)
    #svm = kernel_SVM(X, y, lam, 'poly', degree)

    # run optimizer
    eta_t = lambda t: (t+1)**(-1)
    a = svm.projected_gradient_descent(max_itr=10000, learning_rate=eta_t)
    
    # convert solution of dual -> solution of primal
    w = svm.dual2primal(a)

    # reserve kernel function for contour
    ker = lambda x: svm.gk(x, sigma)
    #ker = lambda x: svm.poly(x, degree)

    # draw Figure
    bc_plot_kernel(X, y, w, ker)

if __name__ == "__main__":
    main()
