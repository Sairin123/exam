# 💫 Hi 👋, I'm MD SAIRIN SAIDAT

#PROGRAM 01: Data Visualization & Outlier Detection
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing(as_frame=True)
df = data.frame 
num_df = df.select_dtypes(include='number') 

num_df.hist(bins=30, figsize=(12, 8), color='steelblue', edgecolor='black')
plt.suptitle('Histograms of Numerical Features')
plt.show()

num_df.plot(kind='box', subplots=True, layout=(3, 3), figsize=(12, 8))
plt.suptitle('Box Plots of Numerical Features')
plt.show()

Q1 = num_df.quantile(0.25)
Q3 = num_df.quantile(0.75)
IQR = Q3 - Q1

outliers_count = ((num_df < (Q1 - 1.5 * IQR)) | (num_df > (Q3 + 1.5 * IQR))).sum()

print("Outliers Detection:\n", outliers_count)
print("\nDataset Summary:\n", df.describe())

#PROGRAM 2: Correlation Matrix & Pair Plot
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing(as_frame=True).frame

plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f') 
plt.title('Correlation Heatmap')
plt.show()

sns.pairplot(data.sample(500), corner=True, diag_kind='kde', plot_kws={'alpha': 0.5})
plt.suptitle('Pair Plot', y=1.02)
plt.show()

#PROGRAM 3: PCA & Scatter Plot
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

iris = load_iris()

X_pca = PCA(n_components=2).fit_transform(iris.data)

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=X_pca[:, 0], 
    y=X_pca[:, 1], 
    hue=iris.target_names[iris.target],  
    palette=['red', 'green', 'blue']
)

plt.title('PCA on Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()


#PROGRAM 04: Find-S Algorithm
import pandas as pd

def find_s_algorithm(file_path):
    df = pd.read_csv(file_path)
    
    print("Training Data:\n", df)
    
    target_col = df.columns[-1]
    positive_cases = df[df[target_col] == 'Yes'].iloc[:, :-1].values
    
    hypothesis = list(positive_cases[0])
    
    for row in positive_cases[1:]:
        for i in range(len(hypothesis)):
            if hypothesis[i] != row[i]:
                hypothesis[i] = '?'
                
    return hypothesis
    
print("\nThe final hypothesis is:", find_s_algorithm(r"C:\Users\gagan\Downloads\training_data - training_data.csv"))

#PROGRAM 05: KNN Classifier & Decision Boundaries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

np.random.seed(42)
X = np.random.rand(100, 1) 
X_train, X_test = X[:50], X[50:]

y_train = np.where(X_train <= 0.5, 1, 2).flatten()

k_values = [1, 3, 5, 20, 30]
train_results = pd.DataFrame({"X_train": X_train.flatten(), "Actual_Class": y_train})
test_results = pd.DataFrame({"X_test": X_test.flatten()})

plt.figure(figsize=(10, 6))
plt.scatter(X_train, [-1]*len(X_train), c=y_train, marker='s', cmap='bwr')

for i, k in enumerate(k_values):
    preds = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train).predict(X_test)
    test_results[f"Pred_k={k}"] = preds
    
    plt.scatter(X_test, [i]*len(X_test), c=preds, marker='o', cmap='bwr')

plt.axvline(x=0.5, color='black', linestyle='--')
plt.yticks([-1] + list(range(len(k_values))), ["Train Data"] + [f"k={k}" for k in k_values])
plt.title("KNN: Decision Boundaries across different K values")
plt.grid(True)
plt.show()

print("\n========== TRAINING DATA ==========\n", train_results.head())
print("\n========== TEST DATA ==========\n", test_results.head())

#Program 06: LWR with Different Tau Values

import numpy as np
import matplotlib.pyplot as plt

def gaussian_kernel(x, x_q, tau):
    return np.exp(-np.square(x - x_q) / (2 * tau**2))

def lwr(x_train, y_train, x_q, tau):
    X = np.c_[np.ones(len(x_train)), x_train]
    W = np.diag(gaussian_kernel(x_train, x_q, tau))
    theta = np.linalg.pinv(X.T @ W @ X) @ (X.T @ W @ y_train)
    return np.array([1, x_q]) @ theta


np.random.seed(42)
x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(0, 0.2, 100)


plt.figure(figsize=(12, 8))
for tau in [0.1, 0.5, 1, 5]:
    y_pred = [lwr(x, y, xq, tau) for xq in x]
    plt.plot(x, y_pred, label=f'tau={tau}')

plt.scatter(x, y, color='black', label='Training Data', alpha=0.5)
plt.xlabel('X'); plt.ylabel('Y')
plt.title('Locally Weighted Regression (LWR) with Different Tau Values')
plt.legend()
plt.show()

#Program 07: Linear & Polynomial Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline

# Linear Part

boston_df = pd.read_csv('C:/Users/gagan/Downloads/BostonHousing.csv')
X_boston = boston_df[['RM']].values
y_boston = boston_df['MEDV'].values

X_train, X_test, y_train, y_test = train_test_split(X_boston, y_boston, test_size=0.2, random_state=42)

linear_reg = LinearRegression().fit(X_train, y_train)
y_pred = linear_reg.predict(X_test)

plt.figure(figsize=(10, 5))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel('Average Number of Rooms (RM)')
plt.ylabel('Housing Price')
plt.title('Linear Regression on Boston Housing Dataset')
plt.legend()
plt.show()
print(f"MSE (Linear Regression): {mean_squared_error(y_test, y_pred):.4f}")

# Polynomial part

col_names = ['mpg', 'cylinders', 'displacement', 'horsepower',
             'weight', 'acceleration', 'model_year', 'origin']

auto_df = pd.read_csv('C:/Users/gagan/Downloads/auto-mpg.csv', names=col_names, header=0, na_values='?').dropna()

X_auto = auto_df[['horsepower']].values
y_auto = auto_df['mpg'].values

X_train, X_test, y_train, y_test = train_test_split(X_auto, y_auto, test_size=0.2, random_state=42)

poly_model = make_pipeline(PolynomialFeatures(3), StandardScaler(), LinearRegression())
poly_model.fit(X_train, y_train)
y_poly_pred = poly_model.predict(X_test)

X_sorted, y_sorted = zip(*sorted(zip(X_test.flatten(), y_poly_pred)))

plt.figure(figsize=(10, 5))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_sorted, y_sorted, color='red', linewidth=2, label='Predicted')
plt.xlabel('Horsepower')
plt.ylabel('MPG')
plt.title('Polynomial Regression on Auto MPG Dataset')
plt.legend()
plt.show()
print(f"MSE (Polynomial Regression): {mean_squared_error(y_test, y_poly_pred):.4f}")

#PROGRAM 08: Decision Tree Classifier & Visualization

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
data = load_breast_cancer(as_frame=True)
cancer_data = data.frame

print("Features:", data.feature_names)
print("Target classes:", data.target_names)
print("Data shape:", cancer_data.shape)
print(cancer_data)
print(cancer_data.isnull().sum())

# Split features and target
X, Y = cancer_data.iloc[:, :-1], cancer_data.iloc[:, -1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train Decision Tree
dtree = DecisionTreeClassifier(max_depth=2, random_state=42).fit(X_train, Y_train)

# Evaluate
y_pred = dtree.predict(X_test)
print("Accuracy:", accuracy_score(Y_test, y_pred))

# Visualize
plot_tree(dtree, filled=True, feature_names=data.feature_names, class_names=data.target_names)
plt.show()

# Predict on new sample
prediction = dtree.predict(X_test.iloc[[0]])
print(prediction)
print("Benign" if prediction == 1 else "Malignant")

from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

faces = fetch_olivetti_faces(shuffle=True, random_state=42)
X, y = faces.data, faces.target
print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

#Program 10: K-Means Clustering & PCA Visualization

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

data = load_breast_cancer()
x, y = data.data, data.target

print("Data shape:", x.shape)
print("Classes:", data.target_names)

x_scaled = StandardScaler().fit_transform(x)

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = kmeans.fit_predict(x_scaled)
labels_mapped = np.where(clusters == 1, 0, 1)

print("\nConfusion Matrix")
print(confusion_matrix(y, labels_mapped))
print("Accuracy:", accuracy_score(y, labels_mapped))

x_pca = PCA(n_components=2).fit_transform(x_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=250, marker='x', c='red', label='Centroids')
plt.title("K-Means Clustering of Breast Cancer Dataset (PCA-2D)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.grid(True)
plt.show()

