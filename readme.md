# Indian Sign Language (ISL) Recognition

This project provides a web-based application for recognizing Indian Sign Language (ISL) hand gestures using a deep learning model. It includes both a Streamlit-based frontend for user interaction and a FastAPI backend for serving predictions.

## Project Structure

```
ISL_ML/
│
├── ISL_model.h5           # Trained Keras model for ISL recognition
├── requirements.txt       # Python dependencies
├── readme.md              # Project documentation
│
├── Backend/
│   └── backend.py         # FastAPI backend for model inference
│
└── Host/
	 └── host.py            # Streamlit frontend for user interaction
```

## Features

- Upload an image of a hand gesture and get the predicted ISL alphabet.
- Deep learning model (Keras/TensorFlow) for gesture recognition.
- Streamlit web interface for easy use.
- FastAPI backend for scalable inference (optional, see below).

## Setup Instructions

1. **Clone the repository**

	```bash
	git clone <repo-url>
	cd ISL_ML
	```

2. **Create and activate a virtual environment (recommended)**

	```bash
	python3 -m venv .venv
	source .venv/bin/activate
	```

3. **Install dependencies**

	```bash
	pip install -r requirements.txt
	```

4. **Ensure `ISL_model.h5` is present in the project root.**

## Usage

### 1. Streamlit Frontend (Recommended)

Run the following command from the project root:

```bash
streamlit run Host/host.py
```

This will launch a web app where you can upload an image of a hand gesture and see the predicted ISL alphabet.

### 2. FastAPI Backend (Optional)

To run the backend API server:

```bash
uvicorn Backend.backend:app --reload
```

You can then send POST requests to `http://localhost:8000/predict` with an image file to get predictions.

#### Example using `curl`:

```bash
curl -X POST "http://localhost:8000/predict" -F "file=@path_to_image.jpg"
```

## Model Details

- The model expects input images of size 128x128 pixels, RGB format.
- Output is one of the ISL alphabets: A, B, C, D, E, F, G, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Z.

## Requirements

- Python 3.7+
- TensorFlow
- Streamlit
- FastAPI
- Uvicorn
- OpenCV
- Pillow
- NumPy

Install all dependencies using the provided `requirements.txt`.

## Notes

- The Streamlit app loads the model directly and does not require the FastAPI backend to be running.
- The FastAPI backend is useful if you want to decouple the frontend and backend or integrate with other services.

## License

This project is for educational purposes.
