# 💊 Drug Side Effect & Interaction Predictor

This project is an AI-powered web application designed to assist users by predicting potential side effects of medications and identifying harmful drug interactions. It leverages a machine learning model trained on medical data and integrates with the Google Gemini API for advanced analysis and real-time information retrieval.

## ✨ Key Features

- **Multi-Step Prediction Form**: A user-friendly, multi-step interface to gather patient information, including ongoing diseases and current medications.
- **AI-Powered Prediction**: Utilizes the Google Gemini API to analyze a patient's new condition and predict a suitable medication.
- **Interaction Warnings**: Checks for and displays potential negative interactions between the user's current medications and the AI-predicted medicine.
- **Side Effect Lookup**: Dynamically fetches and displays known side effects for both current and predicted medications, using the Gemini API with a local data fallback.
- **Modern UI/UX**: A clean, responsive, and visually appealing interface built with a futuristic design aesthetic.
- **In-Memory Caching**: Caches API responses to improve performance and reduce redundant calls for repeated queries.

## 🛠️ Tech Stack

### Backend
- **Python**: Core programming language.
- **Flask**: Web server framework to handle API requests.
- **Google Gemini API**: For generative AI predictions and data retrieval.
- **Scikit-learn**: For the machine learning model (`MultinomialNB` with `HashingVectorizer`).
- **Pandas**: For data manipulation and processing during model training.

### Frontend
- **HTML5 & CSS3**: For structuring and styling the user interface.
- **JavaScript**: For client-side logic, interactivity, and API communication (`fetch`).

## 🚀 Setup and Installation

Follow these steps to get the application running locally.

1.  **Clone the Repository**
    ```bash
    git clone <your-repository-url>
    cd drug-side-effect-prediction
    ```

2.  **Install Dependencies**
    It's recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Set Up Environment Variables**
    Create a `.env` file in the project's root directory and add your Google Gemini API key:
    ```
    GEMINI_API_KEY="YOUR_API_KEY_HERE"
    ```

4.  **Prepare and Train the Model**
    Navigate to the `backend` directory and run the training scripts. This only needs to be done once.
    ```bash
    cd backend
    python extract_specific_data.py
    python train_model.py
    cd ..
    ```

5.  **Run the Flask Server**
    Navigate to the `app` directory and start the server.
    ```bash
    cd app
    python server.py
    ```
    The application will be available at `http://127.0.0.1:5001`.

---

> **⚠️ Disclaimer**: This is an AI-based assistance system and is **not** a substitute for professional medical advice. Always consult a qualified healthcare provider before making any decisions about your health or medication.