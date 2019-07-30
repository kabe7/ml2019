# ml2019 midterm assignment
Repository for Machine Learning(ART.T458) 

## environment
Anaconda 4.7.5  
Python 3.7.3
numpy 1.16.2 (科学技術計算ライブラリ)
matplotlib 3.0.3 (グラフ描画ライブラリ)


```
conda create -n ml python=3.7
...
conda install numpy matplotlib seaborn
```
seabornは見栄えのために入れています。

## about
### Q1: Logistic Regression
src/logistic_regression.pyで実装
```Python
n, dim = 100, 2 
X, y = bc_linear(n, dim) # dataset

# insert dimension for offset 
X = np.insert(X, 2, 1, axis=1)

# build LR instance
lr = LogisticRegression(X, y, lam)

# run optimizer
w, _ = lr.steepest_gradient_descent(learning_rate=1e-3, max_itr=1000)
# w, _ = lr.gauss_newton()

# draw Figure
bc_plot(X, y, w)
```

### Q3: Support Vector Machine
src/svm.pyで実装
```Python
n = 100 # number of data
X, y = bc_linear(n) # dataset
lam = 1 # hyper-parameter

# build SVM instance
svm = SVM(X, y, lam)

# run optimizer
eta_t = lambda t: 1/(t+1)
a, alphas = svm.projected_gradient_descent(learning_rate=eta_t)

# draw figure: 
bc_plot(X, y, svm.a2w(a))
```

### Q5: kerneled-SVM
src/kernel_svm.pyで実装
```Python
n = 100 # number of data
X, y = bc_nonlinear(n) # dataset
lam, sigma, degree = 1, 0.7, 3 # hyper-parameters

# build kernel_SVM instance
svm = kernel_SVM(X, y, lam, 'gauss', sigma)

# run optimizer
eta_t = lambda t: 1/(t+1)
a = svm.projected_gradient_descent(max_itr=10000, learning_rate=eta_t)

# convert solution of dual -> solution of primal
w = svm.dual2primal(a)

# reserve kernel function for contour
ker = lambda x: svm.gk(x, sigma)

# draw Figure
bc_plot_kernel(X, y, w, ker)
```
