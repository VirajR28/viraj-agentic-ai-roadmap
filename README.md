# From Full Stack Development to Agentic AI

This repository is my learning and portfolio space as I move from Java full stack development toward machine learning and AI.

I am learning by building small projects, working through the fundamentals, and keeping the work honest. The goal is not to pretend I have mastered everything at once. The goal is to learn in public, keep improving, and build a clear record of what I have actually worked on.

## Roadmap

### Layer 1 — Machine Learning Foundations

This is the current area of focus. I am learning the basics of Python data work, supervised learning, text preprocessing, feature engineering, model evaluation, and model persistence.

### Layer 2 — Deep Learning

This is planned next. I want to work with neural networks and frameworks such as PyTorch and TensorFlow.

### Layer 3 — Generative AI

This is the next step after the basics. I want to learn more about LLMs, embeddings, retrieval, and practical LLM application patterns.

### Layer 4 — Agentic AI

This is a future focus area. I want to learn more about agent workflows, tool use, planning, and multi-step reasoning.

### Layer 5 — Production

This will come later. I want to learn how to take AI projects from prototype into something more reliable, testable, and deployment-aware.

## Current project

### Email Spam Classifier

The first project in this repository is the spam classifier in [Layer1_ML_Foundations/W05_spam_classifier](Layer1_ML_Foundations/W05_spam_classifier).

It is a small Python project that uses Pandas, NumPy, scikit-learn, TF-IDF, and Logistic Regression to classify SMS-like messages as spam or not spam. It also includes a simple Streamlit interface, model persistence with joblib, and a basic pytest test suite.

This is a learning project, not a production spam detection system. The dataset is intentionally small, which makes it useful for learning the overall workflow without pretending it is a real-world production model.

**Status:** In progress as a learning project. The current result is based on a small demo dataset.

## Projects

| Project | Layer | Technologies | Status |
|---|---|---|---|
| [Email Spam Classifier](Layer1_ML_Foundations/W05_spam_classifier) | Layer 1 — Machine Learning Foundations | Python, Pandas, NumPy, scikit-learn, TF-IDF, Logistic Regression, Streamlit, pytest | Learning project |

## Repository structure

```text
viraj-agentic-ai-roadmap/
├── README.md
├── Layer1_ML_Foundations/
│   └── W05_spam_classifier/
│       ├── app/
│       ├── data/
│       ├── models/
│       ├── notebooks/
│       ├── src/
│       ├── tests/
│       ├── .gitignore
│       ├── README.md
│       ├── requirements.txt
│       └── train_model.py
├── Layer2_Deep_Learning/
├── Layer3_GenAI/
├── Layer4_Agentic_AI/
├── Layer5_Production/
└── .gitignore
```

The future layers are intentionally left empty for now. I will add projects there as the learning journey continues.

## Learning approach

The structure is simple:

Learn → Build → Understand → Document → Improve

I am trying to build things that make the concepts real. I prefer small, working projects over large, theoretical ones. Each project gives me something concrete to learn from and a record of how my understanding grows over time.

## Future direction

This repository will grow from:

- Machine Learning foundations
- Deep Learning
- Generative AI
- Agentic AI
- Production AI and deployment work

The important part is that each layer represents a real stage of learning, not a claim that I already have all of that experience.

---

**Start date:** June 2026
