# End-to-End Deep Learning Project using LSTM & GRU

 **Next Word Prediction using Shakespeare’s "Hamlet"**

This project showcases an **end-to-end Natural Language Processing (NLP)** workflow that predicts the **next word** in a sentence using **LSTM** and **GRU** Recurrent Neural Networks.  
It covers all stages — from **data preprocessing** to **model training** and **Streamlit deployment**.

 **Live Demo:** [Next Word Predictor App](https://end-to-end-dl-project-using-lstm-and-gru-u48zkz57zgvl4mbfaemir.streamlit.app/)

---

## Project Overview

### Objective
Develop a deep learning model that learns linguistic patterns from *Shakespeare’s "Hamlet"* and predicts the next word in a sequence.

### ⚙️ Workflow
1. **Data Collection:** Load the *Hamlet* text from the NLTK Gutenberg corpus.  
2. **Preprocessing:** Clean text, tokenize, and generate padded input sequences.  
3. **Modeling:** Build both **LSTM** and **GRU** architectures.  
4. **Training:** Train the LSTM model for 60 epochs on CPU.  
5. **Deployment:** Implement an interactive **Streamlit app** for real-time prediction.

---

## Dataset

- **Source:** [NLTK Gutenberg Corpus](https://www.nltk.org/)
- **Dataset Used:** *Shakespeare – Hamlet*  
- **Goal:** Learn sequence relationships between words for next-word prediction.

---

## Model Overview

Two deep learning architectures were developed:

- **LSTM (Long Short-Term Memory)**
- **GRU (Gated Recurrent Unit)**

Each model consists of:
- Embedding layer for vector representation of words  
- Recurrent layers (LSTM or GRU)  
- Dropout layers to prevent overfitting  
- Dense output layer with softmax activation  

**Optimizer:** Adam  
**Loss Function:** Categorical Crossentropy  
**EarlyStopping:** Implemented to avoid overfitting.

---

## Training Summary

| Model | Training Accuracy | Validation Accuracy | Epochs | Remarks |
|:------|:-----------------:|:-------------------:|:------:|:--------|
| LSTM  | ~49% | ~5% | 60 | Limited training due to computational constraints |
| GRU   | – | – | – | Architecture implemented but not fully trained |

> **Note:**  
> The LSTM model was trained for only **60 epochs** on a **CPU-based environment**, which is not sufficient for this type of sequential language modeling.  
> The model’s **low validation accuracy** reflects limited training time and resources, not model design issues.  
> With extended training on GPU or cloud hardware, performance is expected to improve significantly.

---

## Deployment

An **interactive Streamlit web app** allows real-time text prediction.  
Users enter a short phrase, and the model predicts the next most likely word.

 **Try the App:**  
 [Streamlit Live App](https://end-to-end-dl-project-using-lstm-and-gru-u48zkz57zgvl4mbfaemir.streamlit.app/)


---

## Future Improvements

- Retrain both LSTM and GRU models on GPU or cloud environments  
- Extend training beyond 60 epochs for better generalization  
- Add Bidirectional LSTM and Attention Mechanism  
- Experiment with Transformer-based architectures (e.g., GPT, BERT)  
- Expand dataset for richer linguistic diversity  
- Integrate TensorBoard for real-time visualization of training metrics  

---

## Author

**Manel Hjaoujia**  
Master’s Student — *Information Systems Engineering & Data Science*  
**manelhjawjia@gmail.com**

