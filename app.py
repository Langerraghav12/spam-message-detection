import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)


MAX_LEN       = 50
TRUNC_TYPE    = "post"
PADDING_TYPE  = "post"
OOV_TOK       = "<OOV>"
VOCAB_SIZE    = 500
EMBEDDING_DIM = 16
N_LSTM        = 20
DROP_LSTM     = 0.2


model     = None
tokenizer = None


def train_model():
    global model, tokenizer

    print("Loading dataset …")
    messages = pd.read_csv("SMSSpamCollection.csv", sep="\t", names=["label", "message"])

    ham_msg  = messages[messages.label == "ham"]
    spam_msg = messages[messages.label == "spam"]

    ham_msg_df = ham_msg.sample(n=len(spam_msg), random_state=44)
    msg_df     = pd.concat([ham_msg_df, spam_msg]).reset_index(drop=True)

    msg_df["msg_type"] = msg_df["label"].map({"ham": 0, "spam": 1})
    msg_label = msg_df["msg_type"].values

    train_msg, _, train_labels, _ = train_test_split(
        msg_df["message"], msg_label, test_size=0.2, random_state=434
    )

    tokenizer = Tokenizer(num_words=VOCAB_SIZE, char_level=False, oov_token=OOV_TOK)
    tokenizer.fit_on_texts(train_msg)

    train_seq    = tokenizer.texts_to_sequences(train_msg)
    train_padded = np.array(pad_sequences(train_seq, maxlen=MAX_LEN,
                                          padding=PADDING_TYPE, truncating=TRUNC_TYPE))
    train_labels = np.array(train_labels)

    _, test_msg, _, test_labels = train_test_split(
        msg_df["message"], msg_label, test_size=0.2, random_state=434
    )
    test_seq    = tokenizer.texts_to_sequences(test_msg)
    test_padded = np.array(pad_sequences(test_seq, maxlen=MAX_LEN,
                                         padding=PADDING_TYPE, truncating=TRUNC_TYPE))
    test_labels = np.array(test_labels)

    print("Training LSTM model …")
    model = Sequential([
        Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN),
        LSTM(N_LSTM, dropout=DROP_LSTM, return_sequences=True),
        LSTM(N_LSTM, dropout=DROP_LSTM, return_sequences=False),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    early_stop = EarlyStopping(monitor="val_loss", patience=2)
    model.fit(
        train_padded, train_labels,
        epochs=30,
        validation_data=(test_padded, test_labels),
        callbacks=[early_stop],
        verbose=1,
    )

    loss, acc = model.evaluate(test_padded, test_labels, verbose=0)
    print(f"Model ready — Test accuracy: {acc:.4f}")


def predict(message: str):
    seq    = tokenizer.texts_to_sequences([message])
    padded = np.array(pad_sequences(seq, maxlen=MAX_LEN,
                                    padding=PADDING_TYPE, truncating=TRUNC_TYPE))
    score  = float(model.predict(padded, verbose=0)[0][0])
    label  = "SPAM" if score >= 0.5 else "HAM"
    return label, score


# ---------------------------------------------------------------------------
# HTML template (single-file UI)
# ---------------------------------------------------------------------------
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>SMS Spam Detector</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      color: #e0e0e0;
    }

    .card {
      background: rgba(255, 255, 255, 0.05);
      backdrop-filter: blur(10px);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 20px;
      padding: 48px 40px;
      width: 100%;
      max-width: 560px;
      box-shadow: 0 25px 50px rgba(0, 0, 0, 0.4);
    }

    .icon { font-size: 48px; text-align: center; margin-bottom: 12px; }

    h1 {
      text-align: center;
      font-size: 1.8rem;
      font-weight: 700;
      margin-bottom: 6px;
      background: linear-gradient(90deg, #e94560, #0f3460);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }

    .subtitle { text-align: center; font-size: 0.9rem; color: #888; margin-bottom: 36px; }

    label {
      display: block;
      font-size: 0.85rem;
      font-weight: 600;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      color: #aaa;
      margin-bottom: 10px;
    }

    textarea {
      width: 100%;
      height: 120px;
      background: rgba(255, 255, 255, 0.07);
      border: 1px solid rgba(255, 255, 255, 0.15);
      border-radius: 12px;
      padding: 14px 16px;
      font-size: 1rem;
      color: #e0e0e0;
      resize: vertical;
      outline: none;
      transition: border-color 0.2s;
    }

    textarea::placeholder { color: #555; }
    textarea:focus { border-color: #e94560; }

    button {
      margin-top: 20px;
      width: 100%;
      padding: 14px;
      background: linear-gradient(90deg, #e94560, #c62a47);
      border: none;
      border-radius: 12px;
      font-size: 1rem;
      font-weight: 700;
      color: #fff;
      cursor: pointer;
      letter-spacing: 0.05em;
      transition: opacity 0.2s, transform 0.1s;
    }

    button:hover  { opacity: 0.9; }
    button:active { transform: scale(0.98); }
    button:disabled { opacity: 0.5; cursor: not-allowed; }

    #result {
      margin-top: 28px;
      border-radius: 14px;
      padding: 20px 24px;
      display: none;
      animation: fadeIn 0.3s ease;
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(8px); }
      to   { opacity: 1; transform: translateY(0); }
    }

    #result.spam { background: rgba(233,69,96,0.15); border: 1px solid rgba(233,69,96,0.4); }
    #result.ham  { background: rgba(39,174,96,0.15); border: 1px solid rgba(39,174,96,0.4); }

    .result-label { font-size: 1.6rem; font-weight: 800; letter-spacing: 0.1em; }
    #result.spam .result-label { color: #e94560; }
    #result.ham  .result-label { color: #27ae60; }

    .result-desc { font-size: 0.9rem; color: #aaa; margin-top: 4px; }

    .bar-wrap {
      margin-top: 16px;
      background: rgba(255,255,255,0.08);
      border-radius: 999px;
      height: 8px;
      overflow: hidden;
    }

    .bar-fill { height: 100%; border-radius: 999px; transition: width 0.5s ease; }
    #result.spam .bar-fill { background: #e94560; }
    #result.ham  .bar-fill { background: #27ae60; }

    .bar-label {
      display: flex;
      justify-content: space-between;
      font-size: 0.78rem;
      color: #777;
      margin-top: 6px;
    }

    .spinner {
      display: inline-block;
      width: 18px; height: 18px;
      border: 3px solid rgba(255,255,255,0.3);
      border-top-color: #fff;
      border-radius: 50%;
      animation: spin 0.7s linear infinite;
      vertical-align: middle;
      margin-right: 8px;
    }

    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>
  <div class="card">
    <div class="icon">📩</div>
    <h1>SMS Spam Detector</h1>
    <p class="subtitle">Powered by a LSTM neural network</p>

    <label for="msg">Your Message</label>
    <textarea id="msg" placeholder="Paste or type an SMS message here…"></textarea>

    <button id="btn" onclick="classify()">Classify Message</button>

    <div id="result">
      <div class="result-label" id="result-label"></div>
      <div class="result-desc"  id="result-desc"></div>
      <div class="bar-wrap">
        <div class="bar-fill" id="bar-fill" style="width:0%"></div>
      </div>
      <div class="bar-label">
        <span>HAM</span>
        <span>SPAM</span>
      </div>
    </div>
  </div>

  <script>
    async function classify() {
      const msg = document.getElementById('msg').value.trim();
      if (!msg) { alert('Please enter a message first.'); return; }

      const btn = document.getElementById('btn');
      btn.disabled = true;
      btn.innerHTML = '<span class="spinner"></span>Classifying…';

      try {
        const res  = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: msg })
        });
        const data = await res.json();

        const resultEl = document.getElementById('result');
        const labelEl  = document.getElementById('result-label');
        const descEl   = document.getElementById('result-desc');
        const barEl    = document.getElementById('bar-fill');

        const isSpam = data.label === 'SPAM';
        const pct    = (data.score * 100).toFixed(1);

        resultEl.className    = isSpam ? 'spam' : 'ham';
        resultEl.style.display = 'block';

        labelEl.textContent = isSpam ? '🚨 SPAM' : '✅ HAM';
        descEl.textContent  = isSpam
          ? `Spam confidence: ${pct}%`
          : `Ham confidence: ${(100 - data.score * 100).toFixed(1)}%`;

        barEl.style.width = pct + '%';

      } catch (err) {
        alert('Error contacting the server. Is the Flask app running?');
      } finally {
        btn.disabled = false;
        btn.textContent = 'Classify Message';
      }
    }

    document.getElementById('msg').addEventListener('keydown', e => {
      if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) classify();
    });
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/predict", methods=["POST"])
def predict_route():
    data    = request.get_json(force=True)
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "No message provided"}), 400

    label, score = predict(message)
    return jsonify({"label": label, "score": round(score, 4)})


if __name__ == "__main__":
    train_model()
    print("\n🚀  Server running at http://127.0.0.1:5001\n")
    app.run(debug=False, host="127.0.0.1", port=5001)
