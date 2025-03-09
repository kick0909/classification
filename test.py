import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ==================== Image Preprocessing ====================
def load_image(url):
    img = cv2.imread(url, cv2.IMREAD_GRAYSCALE)  
    if img is None:
        raise FileNotFoundError(f"Image not found: {url}")
    img = cv2.resize(img, (64, 64))  
    return img / 255.0  

def load_dataset(directory, label_map):
    images, labels = [], []
    
    for class_name in os.listdir(directory):
        class_path = os.path.join(directory, class_name)
        if not os.path.isdir(class_path):
            continue  

        for image_name in os.listdir(class_path):
            img_path = os.path.join(class_path, image_name)

            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue  

                img = cv2.resize(img, (64, 64)) / 255.0  
                images.append(img)
                labels.append(label_map[class_name])

            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

    return np.array(images), np.array(labels)

# ==================== Convolution and Pooling ====================
def convolution2d(image, kernel):
    m, n = kernel.shape
    y, x = image.shape
    y, x = y - m + 1, x - n + 1
    return np.array([
        [np.sum(image[i:i+m, j:j+n] * kernel) for j in range(x)]
        for i in range(y)
    ])

def pooling(image, kernel_size=(2,2), stride=2):
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
    kernel = np.array([[1, 0, -1], [2,0,-2], [1, 0, -1]])
    kernel2 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
    processed_images = []
    for img in images:
        conv1 = convolution2d(img, kernel)
        relu1 = relu(conv1)
        pool1 = pooling(relu1)

        conv2 = convolution2d(pool1, kernel2)
        relu2 = relu(conv2)
        pool2 = pooling(relu2)

        processed_images.append(pool2.flatten())

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
    def __init__(self, input_dim, hidden_dim, output_dim, reg_lambda=0.01):
        self.W1 = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim)
        self.b2 = np.zeros((1, output_dim))
        self.loss_history = []
        self.reg_lambda = reg_lambda

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return Softmax().predict(self.z2)

    def train(self, X, y, X_val, y_val, epochs=10000, learning_rate=0.02):
        softmax = Softmax()
        
        for epoch in range(epochs):
            probs = self.forward(X)
            loss = softmax.loss(self.z2, y) + self.reg_lambda * (np.sum(self.W1**2) + np.sum(self.W2**2))
            self.loss_history.append(loss)

            dL_dz2 = softmax.diff(self.z2, y)
            dW2 = np.dot(self.a1.T, dL_dz2) + self.reg_lambda * self.W2
            db2 = np.sum(dL_dz2, axis=0, keepdims=True)

            dL_dz1 = np.dot(dL_dz2, self.W2.T) * (self.z1 > 0)
            dW1 = np.dot(X.T, dL_dz1) + self.reg_lambda * self.W1
            db1 = np.sum(dL_dz1, axis=0, keepdims=True)

            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2

            if epoch % 1000 == 0:
                val_loss = softmax.loss(self.forward(X_val), y_val)
                print(f"Epoch {epoch}, Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")

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

X_all = np.concatenate((X_train, X_test), axis=0)
y_all = np.concatenate((y_train, y_test), axis=0)

X_train, X_temp, y_train, y_temp = train_test_split(X_all, y_all, test_size=0.2, stratify=y_all)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp)

X_train_features = extract_features(X_train)
X_val_features = extract_features(X_val)
X_test_features = extract_features(X_test)

print(f"Training: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

nn = NeuralNetwork(X_train_features.shape[1], hidden_dim=64, output_dim=len(class_labels), reg_lambda=0.01)
nn.train(X_train_features, y_train, X_val_features, y_val, epochs=5000, learning_rate=0.02)


# ==================== Evaluate Model ====================
y_pred = nn.predict(X_test_features)
accuracy = np.mean(y_pred == y_test) * 100
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