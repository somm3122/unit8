document.addEventListener('DOMContentLoaded', () => {
    const btn = document.getElementById('predict-btn');
    const message = document.getElementById('message');
    const resultDiv = document.getElementById('result');
    const errorDiv = document.getElementById('error-area');

    btn.addEventListener('click', async () => {
        const text = message.value.trim();
        if (!text) {
            errorDiv.innerText = '⚠️ 메시지를 입력하세요.';
            return;
        }
        errorDiv.innerText = '';

        try {
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text })
            });

            if (!res.ok) {
                const err = await res.json().catch(()=>({error: '서버 오류'}));
                errorDiv.innerText = err.error || '서버 오류가 발생했습니다.';
                return;
            }

            const data = await res.json();
            resultDiv.innerHTML = `<p><strong>결과:</strong> ${data.result}</p><p><small>입력: ${data.input}</small></p>`;
        } catch (e) {
            errorDiv.innerText = '서버와 통신 중 오류가 발생했습니다.';
            console.error(e);
        }
    });
});
