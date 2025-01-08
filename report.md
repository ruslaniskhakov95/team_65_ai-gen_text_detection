# Project Documentation

## Project Structure

The project repository is organized as follows:

```
team_65_ai-gen_text_detection/
│
├── api/                                # Folder for API services and utilities
│   ├── logs/                           # Folder for model logs
│   ├── logs_transformers/              # Folder for transformers model logs
│   ├── api_route.py                    # Router for standard models
│   ├── api_transformers.py             # Router for transformers models
│   ├── main.py                         # Main FastAPI application file
│   ├── util_transformers.py            # Helper functions for transformers models
│   └── utils.py                        # Helper functions for standard models
│
├── streamlit_app/                      # Folder for Streamlit application
│   ├── image/                          # Folder for images (e.g., logos)
│   ├── pages/                          # Pages for Streamlit application
│   │   ├── fit.py                      # Page for uploading training data, analysis, and training models
│   │   ├── inference.py                # Page for loading and unloading models for inference
│   │   ├── predict.py                  # Page for predicting target values
│   │   ├── transformer_inference.py    # Page for loading and unloading transformer models
│   │   └── transformer_predict.py      # Page for predicting target values using transformers
│   ├── utils/                          # Utility functions for Streamlit app
│   │   ├── utils.py                    # Helper functions for API requests
│   │   └── transformer_utils.py        # Helper functions for API requests related to transformers
│   └── app.py                          # Main Streamlit application file
│
├── .development.env                    # Example of .env on your server
├── README.md                           # Project official documentation
├── requirements.txt                    # Project dependencies
└── report.md                           # Detailed project documentation
```

---

## Functional Overview

### API

The `api` folder contains all necessary components for serving machine learning models via FastAPI. It supports both standard and transformer-based models.

#### Key Files:

- **`main.py`**: Entry point for the FastAPI server.
- **`api_route.py`**: Handles routing for standard model APIs, including endpoints for loading, training, and inference.
- **`api_transformers.py`**: Handles routing for transformer-based model APIs.
- **`utils.py`**: Contains helper functions for general model operations.
- **`util_transformers.py`**: Contains helper functions specific to transformer models.

#### Features:

- Load and unload models for inference.
- Train models with user-provided datasets.
- Perform inference and predictions using both standard and transformer-based models.

### Streamlit Application

The `streamlit_app` folder contains the frontend application built with Streamlit. It allows users to interact with the models, perform analysis, and visualize results.

#### Key Components:

- **`pages/`**: Modularized pages for different functionalities:
  - `fit.py`: Facilitates uploading datasets, performing exploratory data analysis (EDA), and training models.
  - `inference.py`: Provides controls for managing model loading and unloading during inference.
  - `predict.py`: Enables users to input data and get predictions from standard models.
  - `transformer_inference.py`: Manages transformer model inference lifecycle.
  - `transformer_predict.py`: Handles prediction tasks using transformer-based models.
- **`app.py`**: Main entry point for the Streamlit app.
- **`utils/`**: Utility scripts to streamline API interaction from the Streamlit frontend.

#### Features:

- Upload and analyze training datasets.
- Train models and visualize their performance.
- Load/unload models and perform real-time predictions.
- Support for both standard and transformer-based models.

---

## User Instructions

### Prerequisites

1. Install Python (version 3.8 or higher recommended).
2. Install the required dependencies using:
   ```bash
   pip install -r requirements.txt
   ```

### Running the API

1. Navigate to the `api` folder.
2. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```
3. The API documentation will be available at `http://127.0.0.1:8000/api/openapi`.

### Running the Streamlit Application

1. Navigate to the `streamlit_app` folder.
2. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Open the provided URL in your browser to interact with the application.

### Using the `.env` File

- The `.development.env` file contains environment variables needed to configure the application.
- Copy this file to `.env` and modify the values as per your setup.
- Key variables include:
  - `ROBERTA_PATH`: Path to the RoBERTa model.
  - `DEBERTA_PATH`: Path to the DeBERTa model.
  - `DISTILLBERTA_PATH`: Path to the DistilBERTa model.
  - `API_URL`: URL for connecting to the FastAPI server.

Ensure the `.env` file is placed in the root directory of the project before running the application.

### Workflow Overview

1. **Upload Data**: Use the `Fit` page in Streamlit to upload datasets and perform basic analysis.
2. **Train Models**: Configure training parameters and initiate training from the `Fit` page.
3. **Inference**: Load trained models for inference using the `Inference` or `Transformer Inference` pages.
4. **Make Predictions**: Use the `Predict` or `Transformer Predict` pages to input data and obtain predictions.

## Screenshots
### Start page

<img width="1393" alt="Снимок экрана 2025-01-08 в 14 58 17" src="https://github.com/user-attachments/assets/50588815-5da5-451e-bca3-9978f6033503" />

### Fitting model

<img width="1399" alt="Снимок экрана 2025-01-08 в 15 01 15" src="https://github.com/user-attachments/assets/737bea56-8ff1-41e4-ad56-33c98a6d5090" />

### Descriptive stats

<img width="1321" alt="Снимок экрана 2025-01-08 в 15 02 32" src="https://github.com/user-attachments/assets/752ff388-7af4-4707-b6c3-99c90893d146" />

<img width="1393" alt="Снимок экрана 2025-01-08 в 15 03 24" src="https://github.com/user-attachments/assets/cabdbf3c-6d75-437b-a6a1-004aefbba25a" />

<img width="1399" alt="Снимок экрана 2025-01-08 в 15 04 26" src="https://github.com/user-attachments/assets/2f0b85a7-4741-41e0-b06d-9ba87c9bea25" />

### Model fitted + learning curves

<img width="1354" alt="Снимок экрана 2025-01-08 в 15 06 13" src="https://github.com/user-attachments/assets/24ae5e65-8594-4e13-9407-18c9861c6a72" />

### Model loaded for inference

<img width="1398" alt="Снимок экрана 2025-01-08 в 15 07 00" src="https://github.com/user-attachments/assets/eb5c2191-99e1-408b-9b70-52ef18757991" />

### Random text prediction

<img width="1393" alt="Снимок экрана 2025-01-08 в 15 08 57" src="https://github.com/user-attachments/assets/cce0566c-bb8f-4191-868b-e405b5dd459c" />

### Corpus text prediction (via file upload)

<img width="1393" alt="Снимок экрана 2025-01-08 в 15 10 08" src="https://github.com/user-attachments/assets/d21e44ae-f7f7-46a8-b7de-9f60db538fcf" />


