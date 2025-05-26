# Compound-Facial-Emotion-Recognition-Analysis-GUI

This project implements a GUI-based system for detecting **compound facial emotions** using machine learning and deep learning techniques. The system includes data preprocessing, model training, and a MATLAB GUI for real-time emotion recognition via webcam.

## 🔧 Features

- 📊 **Data Preprocessing**: Normalization and dimensionality reduction for efficient training.
- 🧠 **CNN-based Emotion Classification**: Custom-trained CNN to classify basic and compound emotions.
- 🎛️ **MATLAB GUI**: Interactive GUI (`EMOTION_DETECTION_GUI_APP.m`) for real-time detection using webcam input.
- 📈 **Performance Metrics**: Visualizations include training accuracy, confusion matrix, and confidence bars.
- 📂 **Model Files**: Includes pre-trained `.mat` models for immediate testing.

---

## 🗂️ File Structure

| File Name | Description |
|----------|-------------|
| `EMOTION_DETECTION_GUI_APP.m` | MATLAB GUI script for real-time webcam-based emotion detection |
| `EmotionRecognitionModel.mat` | Trained CNN model |
| `EmotionRecognitionModel2.mat` | Alternative or secondary trained model |
| `compound DATA.ipynb` | Jupyter notebook for dataset handling and preprocessing |
| `compound_emotion_cleaned.mat` | Cleaned and processed dataset |
| `data normalization,reduction.ipynb` | Feature scaling and PCA or other reduction technique |
| `model train.mlx` | MATLAB live script for training the emotion recognition model |
| `accuracy.jpg`, `confusion.jpg`, `training chart.jpg` | Visualizations of model performance |

---

## ▶️ How to Run

### 📌 Prerequisites
- MATLAB (Recommended version: R2021a or later)
- Deep Learning Toolbox
- Python (for Jupyter notebooks if needed)

### 🧪 Running the GUI
1. Open `EMOTION_DETECTION_GUI_APP.m` in MATLAB.
2. Run the script.
3. Ensure webcam is connected.
4. The GUI will display live webcam feed with predicted emotion and confidence.

---

## 📊 Model Insights

- **Accuracy** and **Confusion Matrix** images are included for performance overview.
- Compound emotions like *Happy-Surprise*, *Anger-Disgust* are supported.
- Confidence levels and emoji feedback help visualize predictions.

---

## 📬 Future Improvements

- Add support for more nuanced compound emotions.
- Incorporate facial landmark visualization.
- Expand dataset with more diverse samples.

---

## 📜 License

This project is for academic and research purposes. Please cite appropriately if used in publications.

---

## 👤 Author

Developed by [ayushsingh08-ds](https://github.com/ayushsingh08-ds)
