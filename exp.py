import torch

# 示例矩阵
A = torch.randn(4096,4096)  # 一个 6x4 的矩阵
r = 32  # 目标秩

# SVD 分解
U, S, Vh = torch.linalg.svd(A, full_matrices=False)

# 截断到 rank-r
U_r = U[:, :r]
S_r = S[:r]
V_r = Vh[:r, :]

# 取平方根方便分解成两个矩阵
S_r_sqrt = torch.sqrt(S_r)

# 构造两个矩阵
M1 = U_r * S_r_sqrt.unsqueeze(0)   # m x r
M2 = (S_r_sqrt.unsqueeze(1) * V_r) # r x n

# 验证近似效果
A_approx = M1 @ M2
# print("原矩阵 A:\n", A)
# print("低秩近似 A_approx:\n", A_approx)
print("误差:", torch.norm(A - A_approx).item())
