import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 1. 生成假数据 (20个样本，每个样本256个特征)
X = np.random.rand(20, 256)
y = np.random.rand(20, 256)
z = np.random.rand(20, 256)
colors = np.repeat(np.arange(20), 3)

# 2. 将 X, y, z 的每个样本视为一个独立的数据点
data = np.vstack((X, y, z))  # 合并三个数据集，形成 (60, 256)

# 3. 为每个样本分配一个独特的颜色，基于索引
colors = np.arange(60)  # 创建一个长度为 60 的数组，表示每个样本的索引

# 4. 初始化PCA模型，降维到2维
pca = PCA(n_components=2)

# 5. 对数据进行降维
data_pca = pca.fit_transform(data)

# 6. 打印降维后的数据形状
print("Original data shape:", data.shape)
print("Transformed data shape:", data_pca.shape)

# 7. 可视化降维后的数据，并根据索引着色
scatter = plt.scatter(data_pca[:, 0], data_pca[:, 1], c=colors, cmap='tab20', edgecolor='k', s=100)

# 8. 设置图形的标签和标题
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA of Data with Unique Colors')

# 9. 显示颜色条
plt.colorbar(scatter, label='Sample index')

# 10. 显示图形
plt.show()