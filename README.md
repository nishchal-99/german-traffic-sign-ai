# German Traffic Sign AI 🚦

An end-to-end computer vision project that classifies German traffic signs using deep learning, evaluates model performance, analyzes failures, and provides explainable predictions through a Streamlit app.

## Project Overview

This project uses the German Traffic Sign Recognition Benchmark (GTSRB) to build a traffic sign recognition system across 43 classes.

The final app allows users to upload a traffic sign image and receive:

- predicted traffic sign class
- confidence score
- top 3 predictions
- Grad-CAM explanation showing where the model focused

## Dataset

Dataset: German Traffic Sign Recognition Benchmark (GTSRB)  
Official source: https://benchmark.ini.rub.de/gtsrb_dataset.html  
Official archive: https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html

## Workflow

1. Dataset exploration
2. Image preprocessing
3. Baseline CNN training
4. Improved CNN with augmentation, BatchNorm, Dropout
5. Official test evaluation
6. Failure analysis
7. Grad-CAM explainability
8. Streamlit app deployment

## Results

| Model                  | Validation Accuracy | Official Test Accuracy |
| ---------------------- | ------------------: | ---------------------: |
| Baseline CNN           |              99.13% |                 96.00% |
| Improved CNN           |              99.78% |                 98.03% |
| MobileNetV2 Experiment |              96.03% |           Not selected |

The improved CNN was selected for deployment.

## Key Findings

• The dataset contains class imbalance.
• Image dimensions vary, requiring resizing before training.
• Baseline CNN performed strongly but struggled with blurry, dark, and visually similar signs.
• Improved CNN increased official test accuracy from 96.00% to 98.03%.
• Grad-CAM showed that clean predictions generally focus on meaningful traffic sign regions.
• Remaining failures are mostly caused by low-light, blur, low contrast, and ambiguous triangular warning signs.

## Tech Stack

• Python
• TensorFlow / Keras
• OpenCV
• NumPy
• Pandas
• Scikit-learn
• Matplotlib
• Streamlit

## Run Locally

git clone https://github.com/YOUR_USERNAME/german-traffic-sign-ai.git
cd german-traffic-sign-ai
pip install -r requirements.txt
streamlit run app/app.py

## App Features

• Upload traffic sign image
• Predict one of 43 GTSRB classes
• Show confidence score
• Show top 3 predictions
• Display Grad-CAM explanation

## Citation:

```bibtex
@INPROCEEDINGS{Stallkamp-IJCNN-2011,
  author={Johannes Stallkamp and Marc Schlipsing and Jan Salmen and Christian Igel},
  booktitle={The 2011 International Joint Conference on Neural Networks},
  title={The German Traffic Sign Recognition Benchmark: A multi-class classification competition},
  year={2011},
  pages={1453-1460}
}

```
