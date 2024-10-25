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
from keras_uncertainty.layers import stochastic_layers, SamplingSoftmax
from keras_uncertainty.models import DeepEnsembleClassifier, TwoHeadStochasticRegressor
from keras_uncertainty.models import DisentangledStochasticClassifier, StochasticClassifier
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
    x = stochastic_layers.StochasticDropout(dropout)(x)
    res = x + inputs

    # Feed Forward Network
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation='relu')(x)
    x = stochastic_layers.StochasticDropout(dropout)(x)
    x = Dense(inputs.shape[-1], activation='relu')(x)  # Changed to 'relu' since it is internal
    return x + res

def build_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0, num_classes=2, num_samples=100):
    inputs = Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = GlobalAveragePooling1D(data_format='channels_first')(x)
    for dim in mlp_units:
        x = Dense(dim, activation='relu')(x)
        x = stochastic_layers.StochasticDropout(mlp_dropout)(x)
    #outputs = Dense(num_classes, activation='softmax')(x)  # Changed to 'softmax' for multi-class classification
    
    logit_mean = Dense(num_classes, activation="linear")(x)
    logit_var = Dense(num_classes, activation="softplus")(x)
    probs = SamplingSoftmax(num_samples=num_samples, variance_type="linear_std")([logit_mean, logit_var])

    train_model = Model(inputs, probs, name="train_model")
    pred_model = Model(inputs, [logit_mean, logit_var], name="pred_model")

    return train_model, pred_model#Model(inputs, outputs)

input_shape = X_train.shape[1:]  # input shape (num_channels, sequence_length)

def create_and_train_model(input_shape, num_classes=2):
    train_model, pred_model = build_model(
        input_shape,
        head_size=256,
        num_heads=4,
        ff_dim=4,
        num_transformer_blocks=4,
        mlp_units=[128],
        dropout=0.1,
        mlp_dropout=0.1,
        num_classes=num_classes, 
        num_samples=50
    )
    train_model.summary()
    train_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return train_model, pred_model

def uncertainty(probs):
    return numpy_entropy(probs, axis=-1)

# Apply Ensemble here
num_models = 2
models = []
histories = []
temp_pred_model = []

for _ in range(num_models):
    train_model, pred_model = create_and_train_model(input_shape, num_classes=2) #, history
    models.append(train_model)
    temp_pred_model = pred_model

ensemble = DeepEnsembleClassifier(models=models)
ensemble.fit(X_train, y_train, verbose=2, epochs=12, batch_size=64)
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







NUM_SAMPLES = 50
fin_model = DisentangledStochasticClassifier(temp_pred_model, epi_num_samples=NUM_SAMPLES)
#domain = [X_val]
pred_mean, pred_ale_std, pred_epi_std = fin_model.predict(X_val, batch_size=64)

ale_entropy = uncertainty(pred_ale_std)
epi_entropy = uncertainty(pred_epi_std)
print("pred_mean")
print(pred_mean)
print("ale_entropy")
print(ale_entropy)
print("epi_entropy")
print(epi_entropy)

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

# Plot the training history of all models in the ensemble.
#plot_training_history(histories)

def main():

    print("done")

if __name__ == "__main__":
    main()