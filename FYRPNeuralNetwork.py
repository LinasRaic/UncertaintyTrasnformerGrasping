import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#from tensorflow.python.trackable import base as trackable
#from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
#from tensorflow.keras.models import Model
from tensorflow import keras 
from keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D 
from keras.models import Model
import keras_uncertainty as ku
from keras_uncertainty.models import DeepEnsembleClassifier
from keras_uncertainty.models import DisentangledStochasticClassifier
from keras_uncertainty.utils import numpy_entropy
#from keras_uncertainty.losses import regression_gaussian_nll_loss
from sklearn.model_selection import train_test_split
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print(tf.config.list_physical_devices('GPU'))
# Load segments and labels FOR SINGLE participant


# Load segments and labels for the first participant
X = np.load('X_segments_1.npy', allow_pickle=True)
y = np.load('y_labels_1.npy')

# Perform the train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Network
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1], activation='relu')(x)  # Changed to 'relu' since it is internal
    return x + res

def build_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0, num_classes=2):
    inputs = Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = GlobalAveragePooling1D(data_format='channels_first')(x)
    for dim in mlp_units:
        x = Dense(dim, activation='relu')(x)
        x = Dropout(mlp_dropout)(x)
    outputs = Dense(num_classes, activation='softmax')(x)  # Changed to 'softmax' for multi-class classification
    return Model(inputs, outputs)

input_shape = X_train.shape[1:]  # input shape (num_channels, sequence_length)

def create_and_train_model(input_shape, num_classes=2):
    model = build_model(
        input_shape,
        head_size=256,
        num_heads=4,
        ff_dim=4,
        num_transformer_blocks=4,
        mlp_units=[128],
        dropout=0.1,
        mlp_dropout=0.1,
        num_classes=num_classes
    )
    model.summary()
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #history = model.fit(
    #    X_train, y_train,
    #    validation_data=(X_val, y_val),
    #    epochs=10,
    #    batch_size=32
    #)
    return model#, history

def uncertainty(probs):
    return numpy_entropy(probs, axis=-1)

# Apply Ensemble here
num_models = 5
models = []
histories = []

for _ in range(num_models):
    model = create_and_train_model(input_shape, num_classes=2) #, history
    models.append(model)
    #histories.append(history)

ensemble = DeepEnsembleClassifier(models=models)
ensemble.fit(X_train, y_train, verbose=2, epochs=10, batch_size=32)
preds = ensemble.predict(X_val)
print("predictions:")
print(preds)
print("ground truth:")
print(y_val)
entropy = uncertainty(preds)
print("uncertainty:")
print(entropy)

# Convert prediction probabilities to predicted classes
predicted_classes = np.argmax(preds, axis=1)

# Compare predicted classes with ground truth
correct_predictions = predicted_classes == y_val

# Create a new array indicating correct (True) and wrong (False) predictions
correct_wrong_array = np.where(correct_predictions, 'Correct', 'Wrong')

uncertainty_percentages = np.round(entropy * 100, 2)
uncertainty_percentages[uncertainty_percentages < 0] = 0
# Identify wrong predictions when uncertainty < 10%
wrong_predictions_low_uncertainty = np.logical_and(~correct_predictions, uncertainty_percentages < 10)
count_wrong_low_uncertainty = np.sum(wrong_predictions_low_uncertainty)

# Identify correct predictions when uncertainty > 30%
correct_predictions_high_uncertainty = np.logical_and(correct_predictions, uncertainty_percentages > 30)
count_correct_high_uncertainty = np.sum(correct_predictions_high_uncertainty)

# Print the results
print(f"Number of wrong predictions with uncertainty < 10%: {count_wrong_low_uncertainty}")
print(f"Number of correct predictions with uncertainty > 30%: {count_correct_high_uncertainty}")

print("Predicted Classes: ", predicted_classes)
print("Ground Truth: ", y_val)
print("Correct/Wrong: ", correct_wrong_array)

# Convert entropy to percentages, round to two decimal places, and remove negative values that come due to small values
entropy_percent = np.round(entropy * 100, 2)
entropy_percent[entropy_percent < 0] = 0

# Plot distribution of uncertainty values
plt.figure(figsize=(10, 6))
plt.hist(entropy_percent, bins=20, alpha=0.7, color='b', edgecolor='black')
plt.title('Distribution of Uncertainty Values')
plt.xlabel('Uncertainty (%)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Separate uncertainty values for lateral (1) and palmar (0) grasps
lateral_uncertainty = entropy_percent[y_val == 1]
palmar_uncertainty = entropy_percent[y_val == 0]
'''
# Plot uncertainty for lateral and palmar grasps
plt.figure(figsize=(10, 6))
plt.plot(lateral_uncertainty, 'ro-', label='Lateral Grasp (1)')
plt.plot(palmar_uncertainty, 'bo-', label='Palmar Grasp (0)')
plt.title('Uncertainty for Lateral and Palmar Grasps')
plt.xlabel('Sample Index')
plt.ylabel('Uncertainty (%)')
plt.legend()
plt.grid(True)
plt.show()
'''

#model, _ = create_and_train_model(input_shape, num_classes=2)
fin_model = DisentangledStochasticClassifier(models[0])
#domain = np.c_[X_val.ravel(), y_val.ravel()]
pred_mean, pred_ale_std, pred_epi_std = fin_model.predict(X_val)
ale_entropy = uncertainty(pred_ale_std)
epi_entropy = uncertainty(pred_epi_std)
print(pred_mean)
print(ale_entropy)
print(epi_entropy)
"""
ensemble = DeepEnsembleClassifier(models=models)
history = ensemble.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=32
)

# Evaluate the ensemble model
y_pred_mean, y_pred_std = ensemble.predict(X_val, return_std=True)

"""
def plot_training_history(histories):
    plt.figure(figsize=(14, 6))
    
    for i, history in enumerate(histories):
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']
        
        epochs = range(1, len(loss) + 1)
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs, loss, label=f'Training loss {i+1}')
        plt.plot(epochs, val_loss, label=f'Validation loss {i+1}')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, accuracy, label=f'Training accuracy {i+1}')
        plt.plot(epochs, val_accuracy, label=f'Validation accuracy {i+1}')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

    plt.tight_layout()
    plt.show()

# Plot the training history of all models in the ensemble
plot_training_history(histories)

def main():

    print("done")

if __name__ == "__main__":
    main()