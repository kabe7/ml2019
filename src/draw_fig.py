import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

def bc_plot(X,y,w):
    """
    データとそのpartitioningを表示

    Parameters
    ----------------
    X: np.array(n,2)
        データ
    y: np.array(n)
        ラベル(+1 or -1)
    w: np.array(2) or np.array(3)
        判別パラメータ

    Returns
    ----------------
    Figure: Xの第1,2成分を示した点とw@X = 0となる直線
    +1のラベルは赤, -1のラベルは青で表示
    """
    resolution = 128

    # 描画空間の設定
    base, _ = set_window(X[:,0], X[:,1], resolution)

    # データのプロット
    c = ['r' if label==1 else 'b' for label in y]
    plt.scatter(X[:,0], X[:,1], color=c)

    if len(w) == 2:
        w2 = 0
    elif len(w) == 3:
        w2 = w[2]
    else:
        raise Exception("illegal dimension in parameter w")
    
    # 分割線を引く
    plt.plot(base, - (w2 + base*w[0] - 1) / w[1] , color='r', label='+1')
    plt.plot(base, - (w2 + base*w[0]) / w[1], color='k', label='partition')
    plt.plot(base, - (w2 + base*w[0] + 1) / w[1], color='b', label='-1')
    plt.legend()
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
    kernel: np.array(*, 2) -> np.array(n, *)
        カーネル関数

    Returns
    ----------------
    Figure: Xの第1,2成分を示した点とw@Xの値を示した等高線図
    +1のラベルは赤, -1のラベルは青で表示
    """
    resolution = 128
    
    # 描画空間の設定
    x_base, y_base = set_window(X[:,0], X[:,1], resolution)

    # 等高線のための標高計算
    def calc_contour(x, y, a, n):
        xx, yy = np.meshgrid(x, y)
        positions = np.vstack((xx.flatten(), yy.flatten())).T
        return (a@kernel(positions)).reshape(n,n)

    z = calc_contour(x_base, y_base, a, resolution)

    # 求めた標高をもとに色を塗る
    plt.pcolormesh(x_base, y_base, z, shading='gouraud', cmap='bwr', vmin=-np.max(np.abs(z)), vmax=np.max(np.abs(z)))
    plt.colorbar()

    # データのプロット
    c = ['r' if label==1 else 'b' for label in y]
    plt.scatter(X[:,0], X[:,1], color=c)
    plt.show()

def mc_plot(X, y, W):
    """
    データとそのpartitioningを表示

    Parameters
    ----------------
    X: np.array(n,2)
        データ
    y: np.array(n)
        ラベル, k種類なら0~k-1までの整数値
    W: np.array(2, k) or np.array(3, k)
        判別パラメータ

    Returns
    ----------------
    Figure: Xの第1,2成分を示した点とw@Xの値が最も大きいラベルで領域が薄塗りされた図
    """
    resolution = 256
    dim, k = W.shape
    if dim == 2:
        w = np.insert(W, 2, 0, axis=0)
    elif dim == 3:
        pass
    else:
        raise Exception("illegal dimension in parameter w")
    # 描画空間の設定
    x_base, y_base = set_window(X[:,0], X[:,1], resolution)

    # 色の設定
    cmap = sns.color_palette('Set1', len(np.unique(y)))

    # データのプロット
    c = [cmap[y_i] for y_i in y]
    plt.scatter(X[:,0], X[:,1], color=c, zorder=3)

    #元の分割直線を表示
    for i in range(k):
        plt.plot(x_base, - (W[2,i] + x_base*W[0,i]) / W[1,i], color=cmap[i], alpha= 0.7, label=f"{i} vs other", zorder=2)
    plt.legend()

    # 座標がどのクラスに分類されるかを計算
    def calc_contour(x, y, n):
        xx, yy = np.meshgrid(x, y)
        positions = np.vstack((xx.flatten(), yy.flatten(), np.ones(n*n)))
        return np.argmax(W.T@positions, axis=0).reshape(n,n)
        
    z = calc_contour(x_base, y_base, resolution)
    zcmap = [tuple(0.7 * rgb for rgb in rgbs) for rgbs in cmap]
    plt.contourf(x_base, y_base, z, levels=2, colors=zcmap, alpha=0.3, zorder=1)
    plt.show()

def set_window(x, y, resolution=128):
    """
    描画空間の設定関数

    Parameters:
    ---------
    x, y: 1D array
        x, y軸に表示するデータ
    resolution: positive int(default=128)
        描画空間の分割数

    Returns:
    ---------
    x_base, y_base: np.array(resolution)
        描画空間を格子分割した時のx, y座標の値

    Notes:
    ---------
    各軸は1.2*(データの幅)分表示される
    """
    margin_x, margin_y = 0.1*(max(x)- min(x)), 0.1*(max(y)- min(y))
    x_min, x_max = min(x)-margin_x, max(x)+margin_x
    y_min, y_max = min(y)-margin_y, max(y)+margin_y
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    return np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution)