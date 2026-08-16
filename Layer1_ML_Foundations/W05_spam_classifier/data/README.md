# Dataset

This directory contains data for the spam classifier project.

## Dataset: UCI SMS Spam Collection

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)

**Description:**
- SMS messages in English
- 5,574 messages total
- Binary classification: Spam (747 messages) and Ham (4,827 messages)
- Original data collected by Tiago A. Almeida and José María Gómez Hidalgo
- Released for research and educational purposes

**How to obtain the dataset:**

The full dataset is already included in this directory as `spam.csv` (5,574 messages, `label,text` columns). If the file is ever missing, `load_dataset()` in `src/data_preprocessing.py` will download and rebuild it automatically from the UCI source.

The file contains:
- Column 1 (label): `ham` or `spam`
- Column 2 (text): SMS message content

**Files:**

- `spam.csv` - Full UCI SMS Spam Collection (5,574 messages)

