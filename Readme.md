# ml2019 midterm assignment
Repository for Machine Learning(ART.T458) 

## environment
Anaconda 4.7.5  
Python 3.7.3

```
conda create -n ml python=3.7
...
conda install numpy matplotlib seaborn
```
seabornは見栄えのために入れています。

## about
### Q1: Logistic Regression
src/logistic_regression.pyで実装

### Q2: Lasso

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
a = svm.projected_gradient_descent(max_itr=10000, learning_rate=1e-5)

# convert solution of dual -> solution of primal
w = svm.dual2primal(a)

# reserve kernel function for contour
ker = lambda x: svm.gk(x, sigma)

# draw Figure
bc_plot_kernel(X, y, w, ker)
```
