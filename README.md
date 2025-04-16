# Modular Text Summarization

A modular and well-structured approach to generating abstractive summaries from dialogue-based datasets using state-of-the-art transformer models like BART.

---

## Objective

The objective of this project is to:

- Develop a modular text summarization pipeline.
- Automate the summarization of conversational dialogue using transformer models.
- Provide a clean, extensible codebase for future development and experimentation.
- Evaluate summaries using standard metrics like ROUGE.

---

## Introduction

Text summarization plays a crucial role in condensing large conversations into short, meaningful summaries. In this project, we implement an **abstractive summarization system** using **Hugging Face Transformers**, specifically the **facebook/bart-large-cnn** model. The pipeline is cleanly divided into separate modules for data handling, modeling, and evaluation to encourage best practices and reusability.

---

##  Dataset Description

**Dataset Used:** [SAMSum Dataset](https://huggingface.co/datasets/samsum)

- A collection of informal dialogues with human-written summaries.
- Each data point contains:
  - `dialogue`: A casual conversation between two or more people.
  - `summary`: A concise summary of the conversation.

---

## Project Structure


---

## Design & Architecture

The project follows a modular design:

- **`DataLoader`**: Loads and preprocesses the SAMSum dataset using Hugging Face Datasets.
- **`SummarizationModel`**: Loads a pretrained BART model for summarization tasks.
- **`SummarizationPipeline`**: Handles data flow from loading to prediction.
- **`Evaluator`**: Calculates ROUGE scores to evaluate the generated summaries.

This structure allows for:
- Easy swapping of models or datasets.
- Better maintainability.
- Simplified testing of individual components.

---

## Setup & Installation

### Requirements

- Python 3.7+
- Hugging Face Transformers
- Datasets
- rouge-score
- tqdm

### Setup (Google Colab / Local)

#### Clone the repository:
```bash
git clone https://github.com/your-nasir9586/modular-text-summarization.git
cd modular-text-summarization

### Install dependencies:
pip install -r requirements.txt

Run the pipeline: python main.py

## Note: Formatting and structuring of the text have been assisted using GPT for clarity and consistency.

