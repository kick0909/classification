import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ==================== Image Preprocessing ====================
def load_image(url):
    img = cv2.imread(url, cv2.IMREAD_GRAYSCALE)  # Load grayscale
    if img is None:
        raise FileNotFoundError(f"Image not found: {url}")
    img = cv2.resize(img, (64, 64))  # Resize for standard input
    return img / 255.0  # Normalize pixel values

def load_dataset(directory, label_map, max_images=500):
    images, labels = [], []
    
    for class_name in os.listdir(directory):
        class_path = os.path.join(directory, class_name)
        if not os.path.isdir(class_path):
            continue  

        count = 0
        for image_name in os.listdir(class_path):
            #if count >= max_images:
             #   break
            img_path = os.path.join(class_path, image_name)

            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Skipping corrupt image: {img_path}")
                    continue  

                img = cv2.resize(img, (64, 64)) / 255.0  
                images.append(img)
                labels.append(label_map[class_name])
                count += 1

            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

    images = np.array(images)
    labels = np.array(labels)

    return images, labels


# ==================== Convolution and Pooling ====================
def convolution2d(image, kernel):
    """Perform 2D convolution."""
    m, n = kernel.shape
    y, x = image.shape
    y, x = y - m + 1, x - n + 1
    return np.array([
        [np.sum(image[i:i+m, j:j+n] * kernel) for j in range(x)]
        for i in range(y)
    ])

def pooling(image, kernel_size=(2,2), stride=2):
    """Max pooling operation."""
    m, n = kernel_size
    y, x = image.shape
    y, x = (y - m) // stride + 1, (x - n) // stride + 1
    return np.array([
        [np.max(image[i*stride:i*stride+m, j*stride:j*stride+n]) for j in range(x)]
        for i in range(y)
    ])

# ==================== Activation Functions ====================
def relu(x):
    return np.maximum(0, x)

# ==================== CNN Feature Extractor ====================
def extract_features(images):
    """Apply Convolution + ReLU + Pooling to each image."""
    kernel = np.array([[1, 0, -1], [2,0,-2], [1, 0, -1]])  # Edge detection-like filter
    kernel2 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
    processed_images = []
    for img in images:
        conv1 = convolution2d(img, kernel)   # Convolution
        relu1 = relu(conv1)                  # ReLU Activation
        pool1 = pooling(relu1)               # Max Pooling

        conv2 = convolution2d(pool1, kernel2)   # Convolution
        relu2 = relu(conv2)                  # ReLU Activation
        pool2 = pooling(relu2)               # Max Pooling

        processed_images.append(pool2.flatten())  # Flatten the feature map

    return np.array(processed_images)


# ==================== Softmax and Neural Network ====================
class Softmax:
    def predict(self, X):
        exp_scores = np.exp(X - np.max(X, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def loss(self, X, y):
        num_examples = X.shape[0]
        probs = self.predict(X)
        log_probs = -np.log(np.clip(probs[range(num_examples), y], 1e-7, 1.0))
        return np.mean(log_probs)

    def diff(self, X, y):
        num_examples = X.shape[0]
        probs = self.predict(X)
        probs[range(num_examples), y] -= 1
        return probs / num_examples

class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W1 = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim)
        self.b2 = np.zeros((1, output_dim))
        self.loss_history = []

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = Softmax().predict(self.z2)
        return self.a2

    def train(self, X, y, epochs=1000, learning_rate=0.01, reg_lambda=0.01):
        softmax = Softmax()

        for epoch in range(epochs):
            probs = self.forward(X)
            loss = softmax.loss(self.z2, y)
            self.loss_history.append(loss)

            # Backpropagation
            dL_dz2 = softmax.diff(self.z2, y)
            dW2 = np.dot(self.a1.T, dL_dz2) + reg_lambda * self.W2
            db2 = np.sum(dL_dz2, axis=0, keepdims=True)

            dL_dz1 = np.dot(dL_dz2, self.W2.T) * (self.z1 > 0)
            dW1 = np.dot(X.T, dL_dz1) + reg_lambda * self.W1
            db1 = np.sum(dL_dz1, axis=0, keepdims=True)

            # Update weights
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2

            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

        plt.plot(self.loss_history)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.show()

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)


# ==================== Load Dataset ====================
train_dir = r"/Users/kick/Documents/fruit/fruits-360_dataset_100x100/fruits-360/Training"
test_dir = r"/Users/kick/Documents/fruit/fruits-360_dataset_100x100/fruits-360/Test"

class_labels = sorted(os.listdir(train_dir))
label_map = {label: idx for idx, label in enumerate(class_labels)}

X_train, y_train = load_dataset(train_dir, label_map)
X_test, y_test = load_dataset(test_dir, label_map)

# Combine train and test data for splitting
X_all = np.concatenate((X_train, X_test), axis=0)
y_all = np.concatenate((y_train, y_test), axis=0)

# ==================== Train-Test-Validation Split ====================
# First, split 80% for train + validation and 20% for test
X_train_val, X_test_split, y_train_val, y_test_split = train_test_split(X_all, y_all, test_size=0.2, stratify=y_all)

# Then split the 80% into 80% train and 20% validation (i.e., 64% of the total for training)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, stratify=y_train_val)

# Apply CNN Feature Extraction for train, validation, and test data
X_train_features = extract_features(X_train)
X_val_features = extract_features(X_val)
X_test_features = extract_features(X_test_split)

print(f"Loaded {len(X_train)} training images, {len(X_val)} validation images, and {len(X_test_split)} test images.")
print(f"Feature shape after CNN: {X_train_features.shape}")

# ==================== Train Model ====================
input_dim = X_train_features.shape[1]
hidden_dim = 64
output_dim = len(class_labels)

nn = NeuralNetwork(input_dim, hidden_dim, output_dim)
nn.train(X_train_features, y_train, epochs=10000, learning_rate=0.02)

# ==================== Evaluate Model ====================
y_pred = nn.predict(X_test_features)
accuracy = np.mean(y_pred == y_test_split) * 100
print(f"Test Accuracy: {accuracy:.2f}%")

# ==================== Predict New Image ====================
def predict_new_image(img_path):
    img = load_image(img_path)
    img_resized = cv2.resize(img, (64, 64))
    features = extract_features([img_resized])[0].reshape(1, -1)
    class_scores = nn.forward(features)[0]  # Get softmax probabilities
    pred_class = np.argmax(class_scores)
    class_name = class_labels[pred_class]

    # Display the image with the predicted class
    img_rgb = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title(f"Predicted: {class_name}")
    plt.show()

    # Print class scores
    print("Class Scores:")
    for idx, score in enumerate(class_scores):
        print(f"{class_labels[idx]}: {score:.4f}")

test_img_path = r"/Users/kick/Documents/fruit/fruits-360_dataset_100x100/fruits-360/31dke4F+cTL._AC_UF894,1000_QL80_.jpg"
predict_new_image(test_img_path)