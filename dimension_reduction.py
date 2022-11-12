import pandas as pd
from sklearn.decomposition import PCA
from sklearn import datasets
import plotly.express as px

X = datasets.load_iris()["data"]

# Reducing down to 2D
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

print(pca.explained_variance_ratio_)
print(pca.singular_values_)

df = pd.DataFrame(X_reduced)
df.plot.scatter(x=0, y=1)

# Reducing down to 3D
pca = PCA(n_components=3)
X_reduced = pca.fit_transform(X)

print(pca.explained_variance_ratio_)
print(pca.singular_values_)

df = pd.DataFrame(X_reduced)

fig = px.scatter_3d(df, x=0, y=1, z=2)
fig.show()