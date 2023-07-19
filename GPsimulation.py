import matplotlib.pylab as plt
import numpy as np

np.random.seed(0)


def generate_data(n=10, noise_variance=1e-6):
    np.random.seed(20)
    X = np.random.uniform(-5.0, 5.0, (n, 1))
    X.sort(axis=0)
    y = np.cos(X) + np.random.randn(n, 1) * noise_variance**0.5
    return X, y


# simulating data sets
X_train, y_train = generate_data(10)

X_test = np.arange(-6, 6, 0.05)
y_test = np.cos(X_test)

plt.plot(X_train, y_train, "x", label="Treino")
plt.plot(X_test, y_test, "1", label="Teste")
plt.legend()
plt.show()

noise_var = 1e-6
N, n = len(X_train), len(X_test)


def kernel(x, y, l2=0.1):
    sqdist = np.sum(x**2, 1).reshape(-1, 1) + np.sum(y**2, 1) - 2 * np.dot(x, y.T)
    return np.exp(-0.5 * (1 / l2) * sqdist)


n_samples = 10
mx = np.random.normal(loc=0.0, scale=1.0, size=(N, n_samples))

# calculating similarity matrix
K = kernel(X_train, X_train, l2=0.01)
L = np.linalg.cholesky(K + noise_var * np.eye(N))

# calculating prior
f_prior = np.dot(L, mx)

# Figure 3
plt.scatter(X_train[:, 0], y_train[:, 0], s=100, color="red")
plt.plot(X_train[:, 0], f_prior, alpha=0.6)
plt.show()

# similarity matrices
K = kernel(X_train, X_train, l2=0.1)
K_x = kernel(X_train, X_test.reshape(-1, 1), l2=0.1)
K_xx = kernel(X_test.reshape(-1, 1), X_test.reshape(-1, 1), l2=0.1)

# calculating posterior
L = np.linalg.cholesky(K + noise_var * np.eye(N))
Lk = np.linalg.solve(L, K_x)
mu = np.dot(Lk.T, np.linalg.solve(L, y_train))
L = np.linalg.cholesky(K_xx + noise_var * np.eye(n) - np.dot(Lk.T, Lk))
f_post = mu.reshape(-1, 1) + np.dot(L, np.random.normal(size=(n, n_samples)))

# Figure 4
plt.scatter(X_train[:, 0], y_train[:, 0], s=100, color="red")
plt.plot(X_test, f_post, alpha=0.6)
plt.show()


# Function to make predictions
def posterior(X, Xtest, y, l2=0.1, noise_var=1e-6):
    # compute the mean at our test points.
    N, n = len(X), len(Xtest)
    K = kernel(X, X, l2)
    L = np.linalg.cholesky(K + noise_var * np.eye(N))
    Lk = np.linalg.solve(L, kernel(X, Xtest, l2))
    mu = np.dot(Lk.T, np.linalg.solve(L, y))
    # compute the variance at our test points.
    K_ = kernel(Xtest, Xtest, l2)
    sd = np.sqrt(np.diag(K_) - np.sum(Lk**2, axis=0))
    return (mu, sd)


# Predicting test points
pred_ = posterior(X_train, X_test.reshape(-1, 1), y=y_train, l2=0.1, noise_var=1e-6)
mu = pred_[0][:, 0]
stdv = pred_[1]

# Plotting the results
plt.plot(X_test.reshape(-1, 1)[:, 0], pred_[0][:, 0])
plt.plot(X_test, y_test)
plt.gca().fill_between(X_test, mu - 2 * stdv, mu + 2 * stdv, color="#dddddd")
plt.scatter(X_train, y_train, color="red")
plt.show()
