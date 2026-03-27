# GestureBridge - Sign Language Detection (Version 1)

## 📌 Overview

This project is a real-time sign language detection system that uses a webcam to recognize hand gestures and convert them into text.

---

## ✅ Features

* Real-time hand detection using webcam
* Recognition of numeric gestures (0–9)
* Recognition of Alphabets gesture (A-Z)
* Recognition of 13 common words gesture
* Fast and lightweight system
* Simple interface using OpenCV window

---

## ⚙️ System Requirements

* Python **3.10 (IMPORTANT)**
* Also don't foget to set the environment variable for this python version
* Windows OS
* Webcam

👉 Download Python 3.10 from:
https://www.python.org/downloads/release/python-3100/

---

## 🚀 Setup Instructions (Follow the given commands in the terminal EXACTLY)

### Step 1: Create Virtual Environment

```
py -3.10 -m venv venv
```

### Step 2: Activate Environment

```
venv\Scripts\activate
```

### Step 3: Install Dependencies

```
pip install opencv-python==4.8.1.78 mediapipe==0.10.11 numpy==1.24.4 scikit-learn==1.3.2 matplotlib==3.7.1 pillow==10.0.1 kiwisolver==1.4.5
```

### Step 4: Run the Application

```
python final_system.py
```

---

## ⚠️ Important Notes

* Use a clear background for better accuracy
* Use right hand strictly advisable for best performance 
* Ensure proper lighting conditions
* Close other applications using the camera

---

## 📦 Deliverables

* Source code
* Trained models (.pkl files)
* Setup instructions

---

## 🚀 Future Scope

* GUI-based desktop application
* Additional gestures and words
* Improved accuracy

---

Developed as Version 1 (MVP)
