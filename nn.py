from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import joblib
from sklearn.metrics import classification_report, accuracy_score
# import pickle

# データ取得
iris = load_iris()
x, y = iris.data, iris.target

# 訓練データとテストデータに分割
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)

# モデルのインスタンスを作成(ニューラルネットワーク)
model = MLPClassifier(solver="sgd", random_state=0, max_iter=3000)

# 学習
model.fit(x_train, y_train)
pred = model.predict(x_test)

# 学習済みモデルの保存
joblib.dump(model, "nn.pkl", compress=True)

# 予測精度
print("result: ", model.score(x_test, y_test))
print(classification_report(y_test, pred))
