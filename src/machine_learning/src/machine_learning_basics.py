# import numpy as np
# import matplotlib.pyplot as plt

# # # Rainfall vs. Crop Yield
# # x = np.array([100, 150, 200, 250, 300, 350, 400, 450, 500, 550])  # Rainfall (mm)
# # y = np.array([2.0, 2.5, 3.0, 3.5, 14.0, 4.5, 5.0, 5.5, 6.0, 6.5])  # Crop Yield (tons/ha)

# x = np.array([80, 100, 120, 140, 160, 180, 200, 220, 240, 260])
# y = np.array([35.0, 32.0, 28.5, 25.0, 22.5, 20.0, 18.5, 17.0, 16.0, 15.0])

# # X as single feature
# X = x.reshape(-1, 1)
# y = y.reshape(-1, 1)

# # w = (X^T X)^{-1} X^T y
# XT = X.T
# XTX = XT @ X
# XTX_inv = np.linalg.inv(XTX)
# XTy = XT @ y
# w = XTX_inv @ XTy

# w_optimal = w[0][0]  # Slope only

# # Calculate the regression line
# x_range = np.linspace(x.min(), x.max(), 100)
# y_hat = w_optimal * x_range

# # Save plot
# plt.scatter(x, y, color='blue', label='Data')
# plt.plot(x_range, y_hat, color='red', label=f'y = {w_optimal:.2f}x')
# plt.xlabel('Rainfall (mm)')
# plt.ylabel('Crop Yield (tons/ha)')
# plt.title('Linear Regression: Rainfall vs. Crop Yield')
# plt.legend()
# plt.grid(True)
# plt.savefig('rainfall_yield_regression.png')
# plt.close()

# print(f"Slope (w): {w_optimal:.2f}")


# import numpy as np
# import matplotlib.pyplot as plt

# # Non-linear dataset: y = x^2 + sin(x) + noise
# np.random.seed(0)
# x = np.linspace(-5, 5, 10)
# y = x**2 + np.sin(x) + np.random.normal(0, 1, len(x))

# # Polynomial regression function
# def poly_regression(X, y, degree):
#     XT = X.T
#     XTX = XT @ X
#     XTX_inv = np.linalg.inv(XTX)
#     XTy = XT @ y
#     w = XTX_inv @ XTy
#     return w

# # Prepare polynomial features
# X_1 = np.column_stack([np.ones(len(x)), x])  # Degree 1: [1, x]
# X_2 = np.column_stack([np.ones(len(x)), x, x**2])  # Degree 2: [1, x, x^2]
# X_9 = np.column_stack([x**i for i in range(10)])  # Degree 9: [1, x, ..., x^9]

# # Fit polynomials
# w_1 = poly_regression(X_1, y, 1)
# w_2 = poly_regression(X_2, y, 2)
# w_9 = poly_regression(X_9, y, 9)

# # Generate predictions
# x_range = np.linspace(-5, 5, 100)
# X_range_1 = np.column_stack([np.ones(len(x_range)), x_range])
# X_range_2 = np.column_stack([np.ones(len(x_range)), x_range, x_range**2])
# X_range_9 = np.column_stack([x_range**i for i in range(10)])
# y_pred_1 = X_range_1 @ w_1
# y_pred_2 = X_range_2 @ w_2
# y_pred_9 = X_range_9 @ w_9

# # Save plot
# plt.scatter(x, y, color='blue', label='Data')
# plt.plot(x_range, y_pred_1, color='red', label='Degree 1')
# plt.plot(x_range, y_pred_2, color='green', label='Degree 2')
# plt.plot(x_range, y_pred_9, color='orange', label='Degree 9')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Polynomial Fits: Degrees 1, 2, 9')
# plt.legend()
# plt.grid(True)
# plt.savefig('polynomial_fits.png')
# plt.close()



import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

# Non-linear dataset: y = x^2 + sin(x) + noise
np.random.seed(0)
x = np.linspace(-5, 5, 20)
y = x**2 + np.sin(x) + np.random.normal(0, 1, len(x))

# Polynomial features (degree 9, 10 terms)
X = np.column_stack([x**i for i in range(10)])  # [1, x, x^2, ..., x^9]

# Function for Ridge regression
def ridge_regression(X, y, lambda_reg):
    XT = X.T
    XTX = XT @ X + lambda_reg * np.eye(X.shape[1])  # 10x10 matrix
    XTX_inv = inv(XTX)
    XTy = XT @ y
    w = XTX_inv @ XTy
    return w

# Vary lambda and compute
lambdas = [0.0, 10, 100]
for lam in lambdas:
    w = ridge_regression(X, y, lam)
    y_pred = X @ w
    mse = np.mean((y - y_pred)**2)
    print(f"Lambda: {lam}, MSE: {mse:.2f}, Coefficients: {w.round(2)}")

# Plot for all lambda values
plt.scatter(x, y, color='blue', label='Data')
x_range = np.linspace(-5, 5, 100)
X_range = np.column_stack([x_range**i for i in range(10)])
colors = ['red', 'green', 'orange']
for lam, color in zip(lambdas, colors):
    w = ridge_regression(X, y, lam)
    y_pred = X_range @ w
    plt.plot(x_range, y_pred, color=color, label=f'Lambda={lam}')

plt.xlabel('x')
plt.ylabel('y')
plt.title('L2 Effects on Non-Linear Data (Degree 9)')
plt.legend()
plt.grid(True)
plt.savefig('L2_nonlinear_degree9.png')
plt.close()

