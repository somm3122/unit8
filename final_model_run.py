import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# 1. 데이터 불러오기 및 전처리
df = pd.read_csv('final_200_sample_spam_ham_dataset.csv', encoding='cp949')
df['text'] = df['text'].str.lower().str.strip()

# 2. 벡터화
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

# 3. 학습/테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 최종 모델 학습 (튜닝 결과: C=100)
model = LogisticRegression(C=100, max_iter=1000)
model.fit(X_train, y_train)

# 5. 평가
y_pred = model.predict(X_test)
print(" 최종 평가 결과:")
print(classification_report(y_test, y_pred))

# 6. 혼동 행렬 시각화
disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['ham', 'spam'], cmap='Blues')
disp.ax_.set_title("Confusion Matrix - Final Logistic Model")
plt.show()

# 7. 모델 + 벡터라이저 저장 (.pkl 파일)
final_data = {
    'model': model,
    'vectorizer': vectorizer
}

with open('final_model_with_vectorizer.pkl', 'wb') as f:
    pickle.dump(final_data, f)

print(" 모델과 벡터라이저 저장 완료: final_model_with_vectorizer.pkl")
