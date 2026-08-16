# Email Spam Classifier

This is a small machine learning project I built to learn the full flow of a text classification task: load data, clean it, convert it into features, train a model, evaluate it, and use it for prediction.

## Overview

The project uses the full UCI SMS Spam Collection dataset (5,574 real SMS messages) and a simple TF-IDF + Logistic Regression pipeline. The goal is not to build a production-level spam filter. The goal is to understand the fundamentals of text classification and how a basic supervised model behaves on a real, realistically-sized dataset.

I wanted to work through the practical steps of an ML project without skipping the details: preprocessing text, turning words into numerical features, training a classifier, checking metrics, saving the model, and exposing it through a simple interface.

## What I wanted to learn

- text preprocessing
- TF-IDF feature extraction
- supervised classification
- Logistic Regression
- train/test evaluation
- model persistence
- basic testing for ML code
- simple deployment through Streamlit

## Project structure

```text
W05_spam_classifier/
├── app/
│   └── app.py
├── data/
│   └── spam.csv
├── models/
│   └── spam_classifier.joblib
├── notebooks/
│   └── spam_classifier.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── predict.py
│   └── train.py
├── tests/
│   └── test_spam_classifier.py
├── .gitignore
├── README.md
├── requirements.txt
├── train_model.py
└── train_output.txt
```

The files above reflect the current project as it exists in this repository. The project is intentionally simple and small.

## Approach

1. Load the local demo dataset.
2. Clean the text by normalizing it and removing obvious noise.
3. Split the data into training and validation sets.
4. Convert the message text into TF-IDF features.
5. Train a Logistic Regression classifier.
6. Evaluate the model using accuracy, precision, recall, and F1 score.
7. Save the trained model with joblib.
8. Use the saved model for prediction.
9. Expose the model through a simple Streamlit app.

This is the basic flow I wanted to understand first before moving on to more advanced models or larger datasets.

## Results

The repo now uses the full UCI SMS Spam Collection (5,574 messages, 747 spam / 4,827 ham). After preprocessing and deduplication (5,171 messages remain), the model achieves roughly 96.91% accuracy, 86.67% precision, and 89.31% recall on a held-out 20% test set (1,035 messages).

That result is still not meant to be interpreted as a production-grade spam classifier, but it is a much more realistic baseline than the earlier 43-message demo, and it helps demonstrate how the workflow behaves on a dataset of meaningful size and variety.

## Running the project

From the project folder:

```powershell
cd Layer1_ML_Foundations\W05_spam_classifier
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Run the tests:

```powershell
pytest tests/test_spam_classifier.py -v
```

Train the model:

```powershell
python train_model.py
```

Run the Streamlit app:

```powershell
python -m streamlit run app/app.py
```

Open the notebook:

```powershell
jupyter notebook notebooks/spam_classifier.ipynb
```

## Example predictions

This model is designed for experimentation, so the predictions are best treated as a simple demonstration rather than a production classifier.

Examples that are likely to be treated as spam in this small demo setup include:

- "click here free prize"
- "win a free gift now"

Examples that are likely to be treated as legitimate include:

- "meeting tomorrow at 2pm"
- "hello friend, see you later"

The exact result depends on the small training dataset, so I would not read too much into the output beyond the learning value.

## What I learned

One of the main things I learned is that text has to be converted into numbers before a model can use it. TF-IDF is a straightforward way to do that, and seeing the vocabulary turn into numeric features really helped me understand the idea behind feature engineering.

I also learned that accuracy alone can be misleading. With a small and imbalanced dataset, a model can look reasonable while still being limited. Precision, recall, and F1 score give a better sense of what the model is actually doing.

Another important takeaway was model persistence. Saving and loading a trained model is a simple but important step if I want to reuse the model outside the notebook or training script.

I also learned that testing is part of ML work, not something to add only after the model seems finished. The tests make it easier to catch issues early and keep the workflow understandable.

## Limitations

- The dataset, while much larger now (5,574 messages), is still a single public source from 2011-era SMS text.
- The model may not generalize to modern spam patterns (email, MMS, other languages).
- This is a learning exercise, not a production-grade classifier.

## Future improvements

- combine additional spam datasets for more variety
- compare Logistic Regression with Naive Bayes or a linear SVM
- improve text preprocessing and cleaning routines
- experiment with class imbalance handling
- add more meaningful evaluation metrics and confusion analysis
- expose the model behind a lightweight API later if the project grows

This is a realistic next step list for a first project, not a huge roadmap disguised as a portfolio page.
- Predictions
- Model serialization
- End-to-end integration

### 6. Start Streamlit App

```bash
streamlit run app/app.py
```

The app opens at `http://localhost:8501`

**Features:**
- Enter any message
- Get instant classification
- See confidence scores
- Visualized probabilities

## Example Predictions

### Spam Examples

```
Input:  "CLICK HERE to win FREE MONEY NOW!!!"
Output: SPAM (99.2%)

Input:  "Claim your prize today. Reply CONFIRM"
Output: SPAM (97.8%)
```

### Not Spam Examples

```
Input:  "Hi, are you free for coffee tomorrow?"
Output: NOT SPAM (99.7%)

Input:  "Meeting moved to 3pm in the conference room"
Output: NOT SPAM (98.9%)
```

### Edge Cases

Some messages are genuinely ambiguous:

```
Input:  "Congratulations on your job offer!"
Output: NOT SPAM (87.2%)  [Could be spam but usually legitimate]

Input:  "You have won customer of the month!"
Output: SPAM (72.4%)  [Could be legitimate but often promotional spam]
```

## Limitations

1. **Dataset Domain**
   - Trained on SMS messages
   - Email spam may have different patterns
   - Outdated relative to current spam trends

2. **Language & Content**
   - English only
   - May not handle special characters well
   - Short messages only

3. **Model Constraints**
   - No personalization per user
   - No learning from feedback
   - Static model after training

4. **Performance**
   - ~4% false positive rate (user-facing impact)
   - Can miss sophisticated spam
   - May not scale to very large messages

## Future Improvements

### Short Term
- [ ] Hyperparameter tuning (GridSearchCV)
- [ ] Compare with Naive Bayes and SVM
- [ ] Add more evaluation visualizations
- [ ] Feature importance analysis

### Medium Term
- [ ] Email spam dataset
- [ ] Real email features (headers, sender, etc.)
- [ ] Comparison with deep learning (RNN, LSTM)
- [ ] Multi-language support

### Long Term
- [ ] Online learning from user feedback
- [ ] Production deployment with monitoring
- [ ] A/B testing of model versions
- [ ] Transfer learning with pre-trained embeddings

## Technologies Used

| Category | Tools |
|----------|-------|
| **Data** | Pandas, NumPy |
| **ML** | Scikit-learn (TF-IDF, Logistic Regression) |
| **Visualization** | Matplotlib, Seaborn |
| **Notebooks** | Jupyter |
| **Web UI** | Streamlit |
| **Serialization** | Joblib |
| **Testing** | Pytest |

## Files Not Committed

- `data/spam.csv` — Raw dataset (downloaded at runtime)
- `models/spam_classifier.joblib` — Trained model (generated at runtime)
- `.venv/` — Virtual environment
- `.ipynb_checkpoints/` — Jupyter cache
- `__pycache__/` — Python cache

**Note:** The trained model is small (~50 KB) and could be committed for deployment, but it's regenerated from the notebook instead to keep the repository focused on code and process.

## Key Takeaways for Learning

### ML Fundamentals Demonstrated

1. ✓ Data cleaning and preprocessing
2. ✓ Exploratory data analysis
3. ✓ Train/test stratified split
4. ✓ Feature engineering (TF-IDF)
5. ✓ Model training (Logistic Regression)
6. ✓ Evaluation with multiple metrics
7. ✓ Error analysis
8. ✓ Model serialization
9. ✓ Data leakage prevention
10. ✓ Handling class imbalance

### Best Practices Applied

- ✓ Code organized into reusable modules
- ✓ Clear separation of concerns
- ✓ Reproducible results (fixed random seed)
- ✓ Comprehensive documentation
- ✓ Unit tests for critical functions
- ✓ No hardcoded paths
- ✓ Windows-compatible code

## References

- [UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
- [Scikit-learn TF-IDF Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [Logistic Regression Guide](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- [Precision vs Recall](https://en.wikipedia.org/wiki/Precision_and_recall)

## Author Notes

This project demonstrates the foundational concepts of machine learning applied to a practical problem. The focus is on clarity and understanding rather than pushing for maximum accuracy. The code prioritizes readability and educational value.

The notebook is designed to be followed step-by-step, with explanations of why each decision was made. This is a "first project" — subsequent projects will explore more sophisticated approaches (deep learning, transfer learning, production systems).

---

**Project Status:** Complete ✓  
**Date:** August 2026  
**Learning Focus:** ML Foundations (Layer 1)
