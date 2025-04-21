from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import re
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Layer

app = Flask(__name__)

# Attention Layer Definition
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="attention_weight", shape=(input_shape[-1], input_shape[-1]),
                                 initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="attention_bias", shape=(input_shape[-1],),
                                 initializer="zeros", trainable=True)
        self.u = self.add_weight(name="context_vector", shape=(input_shape[-1], 1),
                                 initializer="glorot_uniform", trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        score = tf.nn.tanh(tf.tensordot(x, self.W, axes=(2, 1)) + self.b)
        attention_weights = tf.nn.softmax(tf.tensordot(score, self.u, axes=(2, 0)), axis=1)
        context_vector = tf.reduce_sum(attention_weights * x, axis=1)
        return context_vector

# Load model and preprocessing objects
model = load_model("disease_prediction_model.h5", custom_objects={'AttentionLayer': AttentionLayer}, compile=False)
with open("preprocessing.pkl", "rb") as f:
    preproc = pickle.load(f)

tokenizer = preproc["tokenizer"]
label_encoder = preproc["label_encoder"]

# Load drug recommendation data
drug_data = pd.read_csv("Disease_Drug_Dataset.csv")
drug_data.columns = drug_data.columns.str.strip()  # Clean headers
disease_drugs = drug_data.groupby("disease")["drug"].apply(list).to_dict()

# Clean text input
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Predict disease and recommend drugs
def predict_disease_and_drug(symptoms):
    cleaned = clean_text(symptoms)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=150)
    prediction = model.predict(padded)
    disease = label_encoder.inverse_transform([np.argmax(prediction)])[0]
    drugs = disease_drugs.get(disease, ["No specific drug found. Please consult a doctor."])
    return disease, drugs

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    disease = None
    drugs = None
    if request.method == 'POST':
        symptoms = request.form['symptoms']
        disease, drugs = predict_disease_and_drug(symptoms)
    return render_template('prediction.html', disease=disease, drugs=drugs)

if __name__ == '__main__':
    app.run(debug=True)