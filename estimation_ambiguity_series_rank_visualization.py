import numpy as np

np.random.seed(0)
N, K = 36, 4

C_true = (np.random.randn(N, N) + 1j * np.random.randn(N, N)) / np.sqrt(2)
H_known = (np.random.randn(N, K) + 1j * np.random.randn(N, K)) / np.sqrt(2)

for R in [1, 3, 5, 8, 9, 10, 15]:
    P_list = [np.diag(np.exp(1j * 2*np.pi * np.random.rand(N))) for _ in range(R)]
    E = np.hstack([P @ H_known for P in P_list])
    Y = np.hstack([C_true @ P @ H_known for P in P_list])
    
    C_est = Y @ np.linalg.pinv(E)
    err = np.linalg.norm(C_est - C_true) / np.linalg.norm(C_true)
    rank_E = np.linalg.matrix_rank(E)
    print(f"R={R:2d}  rank(E)={rank_E:2d}/{N}  err={err:.2e}")