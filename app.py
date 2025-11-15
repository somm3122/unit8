from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

# --- 모델 불러오기 ---
with open('final_model_with_vectorizer.pkl', 'rb') as f:
    data = pickle.load(f)
model = data['model']
vectorizer = data['vectorizer']


# --- 메인 페이지 ---
@app.route('/')
def home():
    return render_template('index.html')


# --- API 엔드포인트: 예측 처리 ---
@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()  # JS에서 보낸 JSON 받기
        text = data.get('message', '')

        if not text.strip():
            return jsonify({'error': '메시지가 비어 있습니다.'}), 400

        # 벡터화 후 예측
        text_vector = vectorizer.transform([text])
        pred = model.predict(text_vector)[0]

        result = '🚨 스팸 메시지로 판단됩니다.' if pred == 'spam' else '✅ 정상 메시지로 판단됩니다.'
        return jsonify({'result': result, 'input': text})

    except Exception as e:
        print(f"예측 중 오류 발생: {e}")
        return jsonify({'error': '서버 내부 오류가 발생했습니다.'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
