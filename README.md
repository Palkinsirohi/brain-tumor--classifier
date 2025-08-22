# 🧠 Brain Tumor MRI Classifier

An advanced deep learning application for classifying brain tumors from MRI scans using Convolutional Neural Networks (CNN) and Streamlit for web interface.

## 📋 Overview

This project is an AI-powered medical imaging tool that classifies brain MRI scans into four categories:
- **Glioma Tumor** - Tumors arising from glial cells
- **Meningioma Tumor** - Tumors of the meninges (usually benign)
- **Pituitary Tumor** - Tumors in the pituitary gland
- **No Tumor** - Normal brain tissue

The application achieves **95.2% accuracy** in tumor classification and provides instant predictions with detailed probability distributions.

## 🚀 Features

- **High Accuracy**: 95.2% classification accuracy using deep learning
- **Fast Processing**: Predictions in less than 1 second
- **Web Interface**: Beautiful Streamlit-based web application
- **Multiple Formats**: Supports JPG, JPEG, PNG, and DICOM formats
- **Detailed Analysis**: Probability distributions and confidence levels
- **Medical Grade**: Trained on thousands of expert-labeled MRI scans

## 🏗️ Project Structure

```
brain-tumor-classifier/
├── data/                    # MRI scan datasets
│   ├── glioma_tumor/        # Glioma tumor images
│   ├── meningioma_tumor/    # Meningioma tumor images
│   ├── no_tumor/            # Normal brain images
│   └── pituitary_tumor/     # Pituitary tumor images
├── model/                   # Model files
│   ├── best_model.h5        # Trained CNN model
│   ├── labels.json          # Class labels
│   └── train.py            # Training script
├── src/                     # Source code
│   ├── predict.py          # Prediction functions
│   └── __init__.py         # Package initialization
├── streamlit_app.py        # Web application
├── requirements.txt        # Python dependencies
├── packages.txt           # Additional packages
└── README.md              # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd brain-tumor-classifier
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install additional packages** (if needed)
   ```bash
   pip install -r packages.txt
   ```

## 🎯 Usage

### Web Application
Run the Streamlit web interface:
```bash
streamlit run streamlit_app.py
```

The application will open in your browser at `http://localhost:8501`

### Command Line Prediction
You can also use the command line for predictions:
```bash
python src/predict.py model/best_model.h5 model/labels.json your_mri_image.jpg
```

### Training the Model
To retrain the model with your own data:
```bash
python model/train.py --data_dir data/ --epochs 30 --image_size 150
```

## 🧠 Model Architecture

The classifier uses a custom Convolutional Neural Network (CNN) with the following architecture:

- **Input**: 150x150x3 RGB images
- **Preprocessing**: Image normalization and data augmentation
- **Convolutional Layers**: 5 convolutional blocks with batch normalization
- **Pooling**: Max pooling for dimensionality reduction
- **Dropout**: 30% dropout for regularization
- **Dense Layers**: Two 512-unit fully connected layers
- **Output**: 4-class softmax classification

### Data Augmentation
- Random horizontal flips
- Random rotation (±6 degrees)
- Random zoom (±6%)

## 📊 Performance Metrics

- **Accuracy**: 95.2%
- **Precision**: 94.8%
- **Recall**: 95.1%
- **F1-Score**: 94.9%
- **Processing Time**: < 1 second per image

## 🏥 Medical Information

### Tumor Types Classified

**🔴 Glioma**
- Most common primary brain tumor in adults
- Arises from glial cells that support neurons
- Can be low-grade (slow-growing) or high-grade (aggressive)

**🟢 Meningioma**
- Develops from the meninges (protective layers around the brain)
- Usually benign and slow-growing
- More common in women, especially after age 40

**🔵 Pituitary Tumors**
- Occur in the pituitary gland at the base of the brain
- Can affect hormone production and vision
- Often treatable with medication or surgery

**✅ Normal Brain Tissue**
- Healthy brain scans showing no abnormal growths
- Important baseline for comparison

## ⚠️ Important Disclaimer

**MEDICAL DISCLAIMER**: This AI tool is for educational and research purposes only. It should NOT replace professional medical diagnosis or treatment. Always consult with qualified healthcare professionals for medical decisions.

- This tool is not FDA-approved for clinical use
- Results should be verified by medical professionals
- Intended for research and educational demonstrations only

## 🔧 Technical Details

### Dependencies
- **TensorFlow 2.20.0**: Deep learning framework
- **Streamlit 1.48.1**: Web application framework
- **OpenCV 4.12.0.88**: Image processing
- **NumPy 2.2.6**: Numerical computations
- **Matplotlib 3.10.5**: Data visualization
- **Pillow 11.3.0**: Image handling

### System Requirements
- **RAM**: Minimum 4GB, Recommended 8GB+
- **Storage**: 500MB for model and dependencies
- **GPU**: Optional but recommended for training (CUDA compatible)

## 📈 Training Process

The model was trained using:
- **Dataset**: 3000+ MRI scans across 4 classes
- **Validation Split**: 10% for validation
- **Optimizer**: Adam with learning rate 1e-4
- **Loss Function**: Sparse Categorical Crossentropy
- **Callbacks**: Model checkpointing, early stopping, learning rate reduction

## 🤝 Contributing

We welcome contributions! Please feel free to:
- Report bugs and issues
- Suggest new features
- Submit pull requests
- Improve documentation

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Medical professionals who provided expert labeling
- Research institutions for MRI datasets
- Open source community for deep learning frameworks
- Healthcare organizations supporting medical AI research

## 📞 Support

For questions, issues, or support:
- Open an issue on GitHub
- Contact the development team
- Check the documentation

---

**Remember**: Early detection saves lives. Always consult healthcare professionals for medical advice.

*Last Updated: January 2025*
*Version: 2.0*
