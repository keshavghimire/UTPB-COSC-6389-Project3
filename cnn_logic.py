import os
import numpy as np
from PIL import Image
from collections import Counter

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

class CNNModel:
    def __init__(self, input_shape=(64, 64, 1), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.initialize_weights()

    def initialize_weights(self):
        self.weights = {
            'conv': np.random.randn(3, 3, 1, 8) * np.sqrt(2 / (3 * 3 * 1)),
            'fc': np.random.randn(7688, self.num_classes) * np.sqrt(2 / 7688)
        }
        self.biases = {'conv': np.zeros((8,)), 'fc': np.zeros((self.num_classes,))}

    def relu(self, x):
        return np.maximum(0, x)

    def max_pooling(self, inputs, pool_size=(2, 2)):
        batch_size, height, width, channels = inputs.shape
        pooled_height = height // pool_size[0]
        pooled_width = width // pool_size[1]
        pooled_output = np.zeros((batch_size, pooled_height, pooled_width, channels))
        for i in range(pooled_height):
            for j in range(pooled_width):
                region = inputs[:, i*pool_size[0]:(i+1)*pool_size[0], j*pool_size[1]:(j+1)*pool_size[1], :]
                pooled_output[:, i, j, :] = np.max(region, axis=(1, 2))
        return pooled_output

    def forward(self, inputs):
        self.inputs = inputs
        self.conv_output = self.convolve(inputs, self.weights['conv']) + self.biases['conv']
        self.conv_output = self.relu(self.conv_output)
        self.pooled_output = self.max_pooling(self.conv_output)
        self.flattened = self.pooled_output.reshape(self.pooled_output.shape[0], -1)
        self.fc_output = np.dot(self.flattened, self.weights['fc']) + self.biases['fc']
        return softmax(self.fc_output)

    def convolve(self, inputs, filters):
        batch_size, height, width, channels = inputs.shape
        filter_height, filter_width, _, num_filters = filters.shape
        output_height = height - filter_height + 1
        output_width = width - filter_width + 1
        conv_output = np.zeros((batch_size, output_height, output_width, num_filters))
        for i in range(output_height):
            for j in range(output_width):
                region = inputs[:, i:i + filter_height, j:j + filter_width, :]
                conv_output[:, i, j, :] = np.tensordot(region, filters, axes=([1, 2, 3], [0, 1, 2]))
        return conv_output

    def compute_loss(self, predictions, labels):
        return -np.sum(labels * np.log(np.clip(predictions, 1e-8, 1.0))) / labels.shape[0]

    def compute_accuracy(self, predictions, labels):
        return np.mean(np.argmax(predictions, axis=1) == np.argmax(labels, axis=1))

    def backward(self, inputs, labels, predictions, learning_rate):
        batch_size = labels.shape[0]
        d_fc_output = predictions - labels
        d_fc_weights = np.dot(self.flattened.T, d_fc_output) / batch_size
        d_fc_biases = np.sum(d_fc_output, axis=0) / batch_size
        d_flattened = np.dot(d_fc_output, self.weights['fc'].T).reshape(self.pooled_output.shape)
        d_conv_output = d_flattened.repeat(2, axis=1).repeat(2, axis=2)
        d_conv_weights = np.zeros_like(self.weights['conv'])
        for i in range(d_conv_output.shape[1]):
            for j in range(d_conv_output.shape[2]):
                region = self.inputs[:, i:i + 3, j:j + 3, :]
                for k in range(d_conv_output.shape[3]):
                    d_conv_weights[..., k] += np.tensordot(region, d_conv_output[:, i, j, k], axes=([0], [0]))
        self.weights['fc'] -= learning_rate * d_fc_weights
        self.biases['fc'] -= learning_rate * d_fc_biases
        self.weights['conv'] -= learning_rate * d_conv_weights
        self.biases['conv'] -= learning_rate * np.sum(d_conv_output, axis=(0, 1, 2)) / batch_size

    def get_layer_info(self):
        layers = [
            ("Conv", f"Filters: {self.weights['conv'].shape[-1]}"),
            ("Pooling", "Size: 2x2"),
            ("FC", f"Neurons: {self.weights['fc'].shape[-1]}")
        ]
        return layers

def preprocess_images(folder_path, image_size=(64, 64)):
    images, labels, class_names = [], [], []
    for root, _, files in os.walk(folder_path):
        if files:
            class_name = os.path.basename(os.path.dirname(root))
            if class_name not in class_names:
                class_names.append(class_name)
            label_index = class_names.index(class_name)
            for filename in files:
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    filepath = os.path.join(root, filename)
                    try:
                        image = Image.open(filepath).convert('L').resize(image_size)
                        images.append(np.array(image) / 255.0)
                        labels.append(label_index)
                    except Exception as e:
                        print(f"Skipping {filepath}: {e}")
    if not images or not labels:
        raise ValueError("No valid images found.")
    images = np.array(images, dtype=np.float32).reshape(-1, image_size[0], image_size[1], 1)
    labels = np.array(labels, dtype=int)
    return images, np.eye(len(class_names))[labels], images.shape[1:]
