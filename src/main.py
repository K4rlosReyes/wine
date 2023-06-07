import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Fix seed for reproducibility
torch.manual_seed(42)

def load_data():
    df_red = pd.read_csv("../data/winequality-red.csv", sep=";")
    df_white = pd.read_csv("../data/winequality-white.csv", sep=";")

    df_red["isred"] = 1
    df_white["isred"] = 0

    df_final = pd.concat([df_red, df_white])

    return df_final


def select_best_features(data, features, target, k=5):
    selector = SelectKBest(f_classif, k=k)
    selector.fit(data[features], data[target])
    feature_scores = selector.scores_
    feature_indices = selector.get_support(indices=True)
    best_features = [features[i] for i in feature_indices]

    plt.figure(figsize=(10, 10))
    plt.bar(features, feature_scores)
    plt.xlabel('Features')
    plt.ylabel('Feature Importance')
    plt.title('Feature Importance Scores')
    plt.xticks(rotation=45)
    plt.show()

    return best_features


def train_models(data, features):
    target = "quality"

    x_train, x_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    models = [
        LogisticRegression(max_iter=1000),
        DecisionTreeClassifier(),
        SVC(),
        RandomForestClassifier()
    ]

    model_names = ['Logistic Regression', 'Decision Tree', 'SVM', 'Random Forest']
    results = []

    for model in models:
        model.fit(x_train_scaled, y_train)
        scores = cross_val_score(model, x_test_scaled, y_test, cv=5)
        results.append(scores.mean())

    # Neural Network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class SimpleNN(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            out = self.fc1(x)
            out = self.relu(out)
            out = self.fc2(out)
            return out

    input_size = len(features)
    hidden_size = 64
    num_classes = 10  # Modify based on your dataset

    x_train_torch = torch.from_numpy(x_train_scaled).float().to(device)
    y_train_torch = torch.from_numpy(y_train.values).long().to(device)
    x_test_torch = torch.from_numpy(x_test_scaled).float().to(device)
    y_test_torch = torch.from_numpy(y_test.values).long().to(device)

    model_nn = SimpleNN(input_size, hidden_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_nn.parameters(), lr=0.001)

    num_epochs = 100

    for epoch in range(num_epochs):
        model_nn.train()
        optimizer.zero_grad()
        outputs = model_nn(x_train_torch)
        loss = criterion(outputs, y_train_torch)
        loss.backward()
        optimizer.step()

    model_nn.eval()
    with torch.no_grad():
        outputs = model_nn(x_test_torch)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y_test_torch).sum().item() / len(y_test_torch)

    model_names.append('Neural Network')
    results.append(accuracy)

    # Plotting the model performance
    plt.figure(figsize=(10, 10))
    plt.bar(model_names, results)
    plt.xlabel('Models')
    plt.ylabel('Mean Accuracy')
    plt.title('Comparison of Model Performance')
    plt.ylim([0, 1])
    plt.xticks(rotation=45)
    plt.show()

    return results


if __name__ == "__main__":
    df = load_data()
    print(df.columns)

    features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                'pH', 'sulphates', 'alcohol', 'isred']

    selected_features = select_best_features(df, features, target="quality", k=7)
    print("Selected Features:", selected_features)

    scores = train_models(df, selected_features)
    print(scores)


