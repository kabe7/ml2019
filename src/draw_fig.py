import numpy as np
import matplotlib.pyplot as plt
import seaborn; seaborn.set()

def bc_plot_2(X,y,w):
    """
    データとそのpartitioningを表示

    Parameters
    ----------------
    X: np.array(n,2)
        データ
    y: np.array(n)
        ラベル(+1 or -1)
    w: np.array(3)
        判別パラメータ

    Returns
    ----------------
    Figure: Xの第1,2成分を示した点とw@X = 0となる直線
    """
    # データのプロット
    x1, x2 = X[:,0], X[:,1]
    c = ['r' if label==1 else 'b' for label in y]
    plt.scatter(x1, x2, color=c)
    # 分割線を引く
    margin_x = 0.1*(max(x1)- min(x1))
    margin_y = 0.1*(max(x2)- min(x2))
    base = np.linspace(min(x1)-margin_x, max(x1)+margin_x,100)
    #plt.xlim(min(x1)-margin_x, max(x1)+margin_x)
    #plt.ylim(min(x2)-margin_y, max(x2)+margin_y)
    plt.plot(base, - (base*w[0] - 1) / w[1] , color='r', label='+1')
    plt.plot(base, - (base*w[0]) / w[1], color='k', label='partition')
    plt.plot(base, - (base*w[0] + 1) / w[1], color='b', label='-1')
    plt.show()

def bc_plot_3(X,y,w):
    """
    データとそのpartitioningを表示

    Parameters
    ----------------
    X: np.array(n,3)
        データ(第1成分は1とする)
    y: np.array(n)
        ラベル(+1 or -1)
    w: np.array(3)
        判別パラメータ

    Returns
    ----------------
    Figure: Xの第2,3成分を示した点とw@X = 0となる直線
    """
    # データのプロット
    x1, x2 = X[:,1], X[:,2]
    c = ['r' if label==1 else 'b' for label in y]
    plt.scatter(x1, x2, color=c)
    # 分割線を引く
    margin_x = 0.1*(max(x1)- min(x1))
    margin_y = 0.1*(max(x2)- min(x2))
    base = np.linspace(min(x1)-margin_x, max(x1)+margin_x,100)
    plt.xlim(min(x1)-margin_x, max(x1)+margin_x)
    plt.ylim(min(x2)-margin_y, max(x2)+margin_y)
    plt.plot(base, - (w[0] + base*w[1] - 1) / w[2] , color='r', label='+1')
    plt.plot(base, - (w[0] + base*w[1]) / w[2], color='k', label='partition')
    plt.plot(base, - (w[0] + base*w[1] + 1) / w[2], color='b', label='-1')
    plt.show()

def bc_plot_kernel(X, y, a, kernel):
    """
    データとそのpartitioningを表示

    Parameters
    ----------------
    X: np.array(n,2)
        データ
    y: np.array(n)
        ラベル(+1 or -1)
    a: np.array(n)
        判別パラメータ
    kernel: np.array(2) -> float
        カーネル関数

    Returns
    ----------------
    Figure: Xの第1,2成分を示した点とw@Xの値を示した等高線図
    """
    n = 100
    x1, x2 = X[:,0], X[:,1]
    
    # 描画空間の設定
    margin_x, margin_y = 0.1*(max(x1)- min(x1)), 0.1*(max(x2)- min(x2))
    x_base, y_base = np.linspace(min(x1)-margin_x, max(x1)+margin_x, n), np.linspace(min(x2)-margin_y, max(x2)+margin_y, n)

    # 等高線を引く
    def calc_contour(x, y, a, n):
        z = np.empty((n,n))
        for i in range(n):
            for j in range(n):
                pos = np.array([x[i], y[j]]).reshape(1,2) 
                z[j,i] = a@kernel(pos)
        
        return z
    
    z = calc_contour(x_base, y_base, a, n)
    plt.contourf(x_base, y_base, z, levels=128, cmap='bwr', vmin=-np.max(np.abs(z)), vmax=np.max(np.abs(z)))
    plt.colorbar()

    # データのプロット
    c = ['r' if label==1 else 'b' for label in y]
    plt.scatter(x1, x2, color=c)

    plt.xlim(min(x1)-margin_x, max(x1)+margin_x)
    plt.ylim(min(x2)-margin_y, max(x2)+margin_y)
    

    
    plt.show()