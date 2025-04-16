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

Insights Document
=================

1. Summary of Findings from Working with the Dataset:
-----------------------------------------------------
- Dataset Characteristics: The dataset used in this project is the **SAMSum dataset**, which consists of informal dialogues and their corresponding human-written summaries. The dataset is small but offers a rich variety of conversation styles, including casual, informal language and mixed speech patterns (slang, abbreviations, etc.).

- Model Performance: The **BART model**, pre-trained on general text summarization tasks, performed reasonably well when tasked with summarizing dialogues. The model was able to generate coherent summaries, retaining the core meaning and structure of the conversation. However, it sometimes struggled with **longer dialogues** and **more complex conversational turns**.

- Challenges in Informal Language: The BART model faced difficulties with informal language, slang, and complex conversational contexts, leading to summaries that sometimes missed the nuance or context of the original conversation. The model also had trouble generating summaries for dialogues that contained a lot of back-and-forth exchanges or non-standard phrasing.


2. Limitations Observed in the Data or Model:
-------------------------------------------
- Data Limitations:
  - **Small Dataset**: The SAMSum dataset is relatively small, which may limit the ability of the model to generalize well to real-world dialogues or conversations outside the dataset's scope.
  - **Informal Language**: The dataset contains **informal language**, abbreviations, and colloquialisms, which are harder for the model to understand, especially when it's been trained on more formal language datasets.
  - **Dialogue Complexity**: Some dialogues in the dataset were highly complex, with a large number of participants, and could involve **ambiguous references** or context-switching. The model struggled with these conversations, often producing summaries that were less clear.

- Model Limitations:
  - **Pretrained Models**: While pretrained models like BART provide a strong baseline, they are not tailored for **specific types of language** or contexts, such as informal dialogues in this case. This could lead to issues in summarization quality when the dialogue structure or language style deviates from what the model was trained on.
  - **Token Limitations**: The token length limitation of models like BART means that long dialogues often get truncated, resulting in loss of critical context or key elements of the conversation.
  - **ROUGE Metrics**: While ROUGE scores were used to evaluate the model's performance, these metrics focus primarily on **n-gram overlap** and do not capture the **semantic meaning** of summaries fully. Thus, a model may score highly on ROUGE but still produce a summary that lacks critical insights or context.


3. Ideas to Explore Next, Given More Time:
-----------------------------------------
- **Fine-Tuning the Model**: One direction for improvement would be to fine-tune the **BART model** on a larger, domain-specific dataset that closely matches the type of informal dialogue in the SAMSum dataset. Fine-tuning on a larger dataset would likely improve the model's ability to handle nuanced dialogues.

- **Incorporating More Advanced Models**: Models like **T5** or **Pegasus**, which are designed specifically for text generation tasks, may perform better than BART for this task. Experimenting with different models could provide better results, especially for informal and context-dependent dialogues.

- **Multimodal Summarization**: Exploring **multimodal summarization**, where additional information such as **tone, emotion, or even visual cues** are incorporated into the dialogue, could help in improving the quality of summaries. This could involve using multimodal models or combining text with audio/video inputs.

- **Human Evaluation**: Incorporating **human evaluation** of the generated summaries could help gain more meaningful insights about the quality of the summaries. ROUGE alone doesn't capture the complexity of summary generation, and human feedback would provide a more comprehensive assessment.

- **Interactive UI**: Building an interactive web interface using **Gradio** or **Streamlit** would allow users to input their own dialogues and generate summaries in real-time. This could also provide additional data for model improvement.

- **Domain-Specific Datasets**: In future work, it would be useful to explore the creation of domain-specific datasets that focus on **customer service**, **medical dialogues**, or **legal conversations**, which have specialized vocabulary and structure. Fine-tuning on such datasets would improve model performance in those areas.

- **Exploring Other Evaluation Metrics**: Apart from **ROUGE**, other evaluation metrics like **BLEU**, **METEOR**, or even **semantic similarity metrics** such as **BERTScore** could be explored to provide a more comprehensive evaluation of the summaries. These metrics would assess not just n-gram overlap but also semantic meaning and context preservation.

Conclusion:
-----------
While the implementation of a text summarization model using the BART architecture performed decently, several improvements can be made by fine-tuning the model, incorporating multimodal information, and exploring new evaluation methods. The limitations of informal language, small datasets, and token restrictions highlight the need for specialized models and domain-specific datasets in the future.

### Setup (Google Colab / Local)

#### Clone the repository:
```bash
git clone https://github.com/your-nasir9586/modular-text-summarization.git
cd modular-text-summarization

### Install dependencies:
pip install -r requirements.txt

Run the pipeline: python main.py

## Note: Formatting and structuring of the text have been assisted using GPT for clarity and consistency.


