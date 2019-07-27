import numpy as np
import matplotlib.pyplot as plt
import seaborn; seaborn.set()
from datasets import *
from draw_fig import bc_plot_2

class SVM:
    """
        Parameters
        ---------------
        X: np.array(n,d)
            データ
        y: np.array(n)
            ラベル
        lam: int
            正則化項の係数

        How to fit
        ----------------
        project_gradient_descentを実行する
    """
    def __init__(self, X, y, lam):
        self.X = X
        self.y = y
        self.lam = lam

    def primal_function(self, a):
        "主問題の目的関数"
        w = self.a2w(a)
        loss = np.clip(1 - self.y * (self.X@w), 0, None)
        return np.sum(loss) + self.lam * w@w

    def dual_Lagrange_function(self, a):
        "双対問題の目的関数"
        half_K = self.y * self.X.T
        K = half_K.T@half_K
        return - 1/(4 * self.lam) * a@K@a + np.sum(a)

    def grad_dLf(self, a):
        half_K = self.y * self.X.T
        K = half_K.T@half_K
        return - 1/(2 * self.lam) * K@a + 1

    def hess_dLf(self,a):
        half_K = self.y * self.X.T
        K = half_K.T@half_K
        return - 1/(2 * self.lam) * K

    def a2w(self, a):
        "双対問題の解 -> 主問題の解 の変換関数"
        w = 1/(2 * self.lam) * np.sum(a * self.y * self.X.T, axis=1)
        return w

    def projected_gradient_descent(self, a0=None, learning_rate=0.01, eps=1e-6, max_itr=10000):
        """
        射影勾配法による双対問題の最適解の求解関数

        Parameters
        ---------------
        X: np.array(n,d)
            データ
        y: np.array(n)
            ラベル
        lam: int
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
        alpha = np.clip(a0, 0, 1) if a0 is not None else np.clip(np.random.normal(size=(n,)), 0, 1)
        prev = self.dual_Lagrange_function(alpha)
        alpha_log = [alpha]
        for itr in range(max_itr):
            alpha = alpha + learning_rate * self.grad_dLf(alpha)
            alpha = np.clip(alpha, 0, 1)
            now = self.dual_Lagrange_function(alpha)
            print(f"iteration {itr}: {prev} -> {now}")
            alpha_log.append(alpha)
            if(abs(prev - now) < eps):
                print(f"iteration stop at {itr}")
                break
            else:
                prev = now
        return alpha, alpha_log

def main():
    n, dim = 100, 2
    lam = n / 4
    X, y = bc_linear(n, dim)
    svm = SVM(X, y, lam)
    
    a, alphas = svm.projected_gradient_descent()
    print(a)

    bc_plot_2(X, y, svm.a2w(a))
    
    itr = len(alphas)
    primals = [svm.primal_function(a) for a in alphas]
    duals = [svm.dual_Lagrange_function(a) for a in alphas]

    # draw figure: primal vs dual
    plt.title("dual v.s. primal (eps=1e-6)")
    plt.xlabel("iteration")
    plt.xticks(np.linspace(0, (itr//1000 + 1) * 1000, (itr//1000 + 1) * 2 + 1))
    plt.ylabel("value of loss function")
    plt.plot(np.arange(itr), primals, label="primal")
    plt.plot(np.arange(itr), duals, label="dual")
    plt.legend()
    plt.show()
    
    # draw figure: diffrenece between primal and dual
    plt.title("primal-dual")
    plt.xlabel("iteration")
    plt.ylabel("difference(log scale)")
    plt.plot(np.arange(itr), [p-d for p, d in zip(primals, duals)])
    plt.yscale('log')
    plt.show()

if __name__ == '__main__':
    main()