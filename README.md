# SMS Spam Classifier — LSTM

A deep-learning SMS spam detector built with TensorFlow/Keras (LSTM) and served via a Flask web app.


---

## Project structure

```
spam_classifier/
├── app.py                    # Flask web app (trains model on startup, serves UI + /predict API)
├── LSTM_Spam_Detection.ipynb # Jupyter notebook with full EDA + training walkthrough
├── SMSSpamCollection.csv     # Dataset (tab-separated: label \t message)
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

Place `SMSSpamCollection.csv` in the same folder as `app.py`.

---

## Run the web app

```bash
python app.py
```

The script will:
1. Load and balance the dataset (down-sample ham to match spam count).
2. Tokenize and pad sequences.
3. Train a two-layer LSTM model (up to 30 epochs with early stopping).
4. Start a Flask server at **👉 http://127.0.0.1:5001**.

Open the URL in your browser, type any SMS message, and click **Classify Message**.

---

## Model architecture

| Layer       | Details                              |
|-------------|--------------------------------------|
| Embedding   | vocab=500, dim=16, input_len=50      |
| LSTM        | units=20, dropout=0.2, return_seq=True |
| LSTM        | units=20, dropout=0.2                |
| Dense       | 1 unit, sigmoid activation           |

Loss: `binary_crossentropy` | Optimizer: `adam`

---

## Dataset

The UCI SMS Spam Collection dataset — 5,572 messages labelled **ham** or **spam**.  
Ham messages are down-sampled to 747 to match the spam count before training.

---

## API

`POST /predict`

```json
// Request
{ "message": "Congratulations! You've won a free prize. Call now!" }

// Response
{ "label": "SPAM", "score": 0.9821 }
```
