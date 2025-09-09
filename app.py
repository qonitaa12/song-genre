from flask import Flask, request, render_template
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model dan tools
model = load_model("genre_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

app = Flask(__name__)
MAX_LEN = 200  # Sesuai waktu training

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['lyrics']
        sequence = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
        pred = model.predict(padded)
        label = label_encoder.inverse_transform([np.argmax(pred)])
        return render_template('index.html', prediction=label[0], input_lyrics=text)

if __name__ == '__main__':
    app.run(debug=True)
