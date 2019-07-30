import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from datasets import *
from draw_fig import *
from logistic_regression import LogisticRegression

def one_vs_all(X, y, lam):
    """
    多値分類の判別器

    Parameters
    --------------------
    X: np.array(n,d)
        データ
    y: np.array(n)
        ラベル(k種類)
    lam: int
        正則化項の係数

    Returns:
    W: np.array(d,k)
        W = (w1, w2, ..., wk)
        各w_iはi番目の要素とその他の要素を分類する判別直線の係数
    """
    labels = np.unique(y)
    X = np.insert(X, 3, 1, axis=1)
    n, d = X.shape
    w = np.empty((d,len(labels)))
    for i, main_label in enumerate(labels):
        label = np.array([1 if y_i == main_label else -1 for y_i in y])
        lr = LogisticRegression(X, label, lam)
        eta_t = lambda t: 1/(t+1)
        w[:,i], _ = lr.steepest_gradient_descent(learning_rate=eta_t, max_itr=1000)
        # bc_plot(X[:,1:3], label, w[1:4,i])
    return w

def main():
    n=100
    X, y = mc_linear(n)    
    w = one_vs_all(X,y,1)
    plt.title("3-class classification")
    mc_plot(X[:,1:3], y, w[1:4,:])

if __name__ == "__main__":
    main()