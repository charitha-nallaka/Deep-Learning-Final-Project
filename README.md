# Final Project Assignment: 1
## Dataset Selection & Problem Definition

### Group Members:
- Charitha Nallaka
- Geethika Sannala

---

## Datasets:

### Dataset 1: Brain Tumor MRI Classification
- **Source**: [Brain Tumor MRI Dataset on Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Description**: This dataset contains 7,023 MRI images categorized into four classes:
  - **Glioma**: A type of tumor that starts in the glial cells of the brain or spine.
  - **Meningioma**: A tumor that forms on membranes covering the brain and spinal cord.
  - **Pituitary Tumor**: A tumor occurring in the pituitary gland, affecting hormone regulation.
  - **No Tumor**: Normal MRI scans without tumors.
- **Problem Definition**:
  - **Goal**: Classify brain MRI images into one of the four categories (Glioma, Meningioma, Pituitary Tumor, or No Tumor).
  - **Problem Type**: Multi-class classification
  - **Target Labels**: {Glioma, Meningioma, No Tumor, Pituitary Tumor}
- **Preprocessing Steps**:
  - Image resizing to a standard dimension
  - Removing extra margins to improve model accuracy
  - Normalization and augmentation techniques may be applied to enhance generalization
  - The dataset is suitable for deep learning frameworks such as TensorFlow or PyTorch
  - CNN-based models (e.g., VGG16, ResNet, EfficientNet) can be used for classification
  - Additional segmentation models may be applied for tumor localization

---

### Dataset 2: Urban Sound Classification
- **Source**: [Urban Sound Dataset on Kaggle](https://www.kaggle.com/datasets/chrisfilo/urbansound8k/data)
- **Description**: This dataset contains 8,732 labeled sound excerpts from urban environments, including various types of noise like street sounds, music, and human voices. The data is categorized into 10 classes based on the sound type.
- **Problem Definition**:
  - **Goal**: Classify urban sounds into one of the predefined categories (e.g., car horn, dog bark, air conditioner, etc.).
  - **Problem Type**: Multi-class classification
  - **Target Labels**: {Air Conditioner, Car Horn, Children Playing, Dog Bark, Drilling, Engine Idling, Gunshot, Jackhammer, Sire, Street Music}
- **Dataset Requirements**:
  - The dataset consists of 8,732 labelled audio clips, evenly distributed across 10 categories
  - Audio clips are 4 seconds long, sampled at 22,050 Hz with a mono channel
  - The dataset is split into training, validation, and testing sets, with files available in `.wav` format
  - Preprocessing steps include converting audio into spectrograms or MFCC features for model input
  - Deep learning frameworks such as TensorFlow or PyTorch can be used for building CNN models to perform sound classification

---

### Dataset 3: Mental Health Sentiment Analysis
- **Source**: [Mental Health Sentiment Analysis Dataset on Kaggle](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health/data)
- **Description**: This dataset is designed for sentiment analysis in the mental health domain, containing textual data labeled into seven distinct mental health categories. The goal is to analyze linguistic patterns and emotional markers that correlate with different mental health conditions.
- **Categories**:
  - **Normal**: Text with no indication of mental distress.
  - **Depression**: Expressions of sadness, hopelessness, or lack of motivation.
  - **Suicidal**: Text indicating thoughts of self-harm or suicidal ideation.
  - **Anxiety**: Expressions of excessive worry, nervousness, or stress-related fear.
  - **Stress**: Text indicating high levels of tension and emotional strain.
  - **Bipolar**: Expressions suggesting mood swings, mania, or depressive episodes.
  - **Personality Disorder**: Text reflecting patterns associated with personality disorders.
- **Problem Definition**:
  - **Goal**: Classify text into one of the seven mental health categories.
  - **Problem Type**: Multi-class text classification
  - **Target Labels**: {Normal, Depression, Suicidal, Anxiety, Stress, Bipolar, Personality Disorder}
- **Dataset Requirements**:
  - The dataset consists of labeled text data categorized into seven mental health conditions
  - Text data requires preprocessing, including tokenization, stopword removal, and vectorization for model input
  - Preprocessing steps include:
    - Text cleaning: Removing special characters, stopwords, and unnecessary whitespace
    - Tokenization: Converting text into individual words or subwords
    - Normalization: Lowercasing, stemming, or lemmatization
    - Feature Engineering: Using TF-IDF, word embeddings (Word2Vec, GloVe), or transformer-based embeddings (BERT)
    - Data Augmentation: Expanding training samples using paraphrasing techniques if needed

---

## How Each Dataset Satisfies the Criteria:

1. **Dataset 1 (Brain Tumor MRI Classification)**:
   - Offers 7,023 labeled MRI images, suitable for deep learning tasks, particularly CNNs. The dataset can be preprocessed through resizing, normalization, and augmentation for optimal performance in automated tumor classification.

2. **Dataset 2 (Urban Sound Classification)**:
   - Contains 8,732 labeled audio clips, evenly distributed across 10 categories, making it compatible with deep learning frameworks. The audio data can be preprocessed into spectrograms or MFCC features for model input, supporting multi-class classification tasks.

3. **Dataset 3 (Mental Health Sentiment Analysis)**:
   - Provides sufficient labeled text data for training a classification model in sentiment analysis for mental health. The dataset is balanced, with well-defined categories for various mental health conditions. Preprocessing and feature engineering are required for effective model training.

---

## Code Repository:
[GitHub Repository for Deep Learning Final Project](https://github.com/charitha-nallaka/Deep-Learning-Final-Project)
