import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from datasets import bc_linear_4
from draw_fig import *

class LogisticRegression:
    """
        Parameters
        --------------------
        X: np.array(n,d)
            データ
        y: np.array(n)
            ラベル
        lam: int
            正則化項の係数
    """
    def __init__(self, X, y, lam):
        self.X = X
        self.y = y
        self.lam = lam

    # 目的関数群
    def J(self, w):
        p = 1/(1 + np.exp(-self.y * (self.X@w)))
        return - np.sum(np.log(p)) + self.lam * w@w

    def grad_J(self,w):
        p = 1/(1 + np.exp(-self.y * (self.X@w)))
        return - np.sum(self.X.T * self.y * (1-p), axis=1) + 2 * self.lam * w

    def hess_J(self,w):
        p = 1/(1 + np.exp(-self.y * (self.X@w)))
        return (p * (1-p) * self.X.T)@self.X + 2 * self.lam 

    def optimize(self, f, impr, w0=None, max_itr=1000, eps=1e-6):
        """
        反復による最適化の型紙

        Parameters
        ----------------------
        f: np.array(d) -> float
            目的関数
        impr: np.array(d), int -> np.array(d)
            パラメータ改善の関数
            w_(t+1) = w_t + impr(w_t, itr)
        w0: np.array(d)
            初期値
        max_itr: Int
            最大反復回数
        eps: float
            反復終了条件

        Returns
        ------------------------
        w: np.array(d) = 判別パラメータ
        ws: List[np.array(d)] = 各反復でのwの値
        """
        _, d = self.X.shape
        if w0 is None:
            w = np.empty(d)
        else:
            w = np.copy(w0)
        prev = f(w)
        f_values = [prev]
        for itr in range(max_itr):
            w = w + impr(w, itr)
            now = f(w)
            print(f"iteration {itr}: {prev} -> {now}", end='\r')
            f_values.append(now)
            if(abs(prev - now) < eps):
                print(f"\niteration stop at {itr}")
                break
            else:
                prev = now
        return w, f_values

    def steepest_gradient_descent(self, w0=None, learning_rate=0.01, max_itr=1000, eps=1e-6):
        if not callable(learning_rate):
            eta = lambda itr: learning_rate
        else:
            eta = learning_rate

        f = self.J
        impr = lambda w, itr: - eta(itr) * self.grad_J(w)
        return self.optimize(f, impr, w0, max_itr, eps)

    def gauss_newton(self, w0=None, max_itr=1000, eps=1e-6):
        f = self.J
        impr = lambda w, itr: - np.linalg.solve(self.hess_J(w), self.grad_J(w))
        return self.optimize(f, impr, w0, max_itr, eps)

def test(n=100, lam=1):
    X, y = bc_linear_4(n)

    # offset用の次元を挿入
    X = np.insert(X, 3, 1, axis=1)

    #w0 = np.empty(5)
    w0 = np.zeros(5)

    # calculation
    lr = LogisticRegression(X, y, lam)
    learning_rate = lambda t: 1/(t+1)
    w, ws_sgd = lr.steepest_gradient_descent(w0, learning_rate=1e-3, max_itr=10000)
    try:
        _, ws_gn = lr.gauss_newton(w0)
    except np.linalg.LinAlgError:
        ws_gn = np.empty(len(ws_sgd))

    # partitioningの確認
    bc_plot(X[:,1:4], y, w[1:4].T)

    # 各手法で求めた最適値との差分を取る
    ws_sgd, ws_gn = [(l-min(l))[:len(l)-1] for l in [ws_sgd, ws_gn]]
    
    plt.title("GD v.s. Newton (eps=1e-6)")
    plt.xlabel("iteration")
    plt.ylabel("difference from optimal")
    plt.yscale("log")
    plt.plot(np.arange(len(ws_sgd)), ws_sgd, label="GD")
    plt.plot(np.arange(len(ws_gn)), ws_gn, label="Newton")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    test()