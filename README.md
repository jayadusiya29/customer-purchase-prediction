# customer-purchase-prediction# Customer Purchase Prediction

This project demonstrates a simple binary classification model using a neural network to predict whether a customer will make a purchase based on two features:

- Website Visit Duration
- Number of Pages Visited

The dataset is synthetically generated, and the model is built using TensorFlow/Keras.

---

## 📊 Dataset

A synthetic dataset with 200 data points is created using NumPy. Each data point consists of:

- `VisitDuration`: A random float between 0 and 1
- `PagesVisited`: A random float between 0 and 1
- `Purchase`: A binary label (0 or 1), determined by whether the sum of the two features is greater than 1.

---

## 🧠 Model Architecture

A simple feedforward neural network:

- Input Layer: 2 nodes (VisitDuration, PagesVisited)
- Hidden Layer: 10 nodes, ReLU activation
- Output Layer: 1 node, Sigmoid activation

Loss function: `binary_crossentropy`  
Optimizer: `adam`  
Metric: `accuracy`

