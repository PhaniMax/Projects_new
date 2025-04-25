# Importing Necessary Modules and Libraries

from eda_utils import display_image_samples, plot_class_distribution
from import_libraries import get_general_utils
from image_processing import load_and_preprocess_images
from import_libraries import get_data_handling_and_viz_libs
from import_libraries import get_core_keras_layers, get_training_components
from import_libraries import get_data_preprocessing_tools, get_sklearn_components, get_keras_utilities

from tensorflow.keras.optimizers.legacy import Adam
import tensorflow as tf




# Data loading and Preprocessing
Path, os = get_general_utils()
dataset_path = Path("contents/dataset/")

SIZE = 64
COLOR_MODE = 'rgb'
infected_data, infected_labels = load_and_preprocess_images(dataset_path/'Parasitized', SIZE, COLOR_MODE)
uninfected_data, uninfected_labels   = load_and_preprocess_images(dataset_path/'Uninfected' , SIZE, COLOR_MODE)




# Combine the datasets and labels
dataset = infected_data   + uninfected_data
labels  = infected_labels + uninfected_labels

print(f"Total images: {len(dataset)}")
print(f"Number of 'Infected' images: {len(infected_data)}")
print(f"Number of 'Uninfected' images:  {len(uninfected_data)}")




# Exploratory data Analysis
np, plt, sns, cv2 = get_data_handling_and_viz_libs()
samples_to_display = 10
random_indices = np.random.choice(len(dataset), samples_to_display, replace=False)
samples = [dataset[i] for i in random_indices]
sample_labels = [labels[i] for i in random_indices]
print(samples[0].dtype)
display_image_samples(samples, sample_labels, sample_size = samples_to_display)
plot_class_distribution(labels)





# Model Architecture
Input, Conv2D, Dense, Flatten, BatchNormalization, Dropout, Model = get_core_keras_layers()
l2, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard = get_training_components()

def conv_block(input_tensor, num_filters):

    x = Conv2D(num_filters, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(input_tensor)
    x = Conv2D(num_filters, (3, 3), activation='relu', padding='same', strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    return x

def dense_block(input_tensor, num_neurons):

    x = Dense(num_neurons, activation='relu', kernel_regularizer=l2(0.001))(input_tensor)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    return x

# Define the input shape
INPUT_SHAPE = (SIZE, SIZE, 3)

# Input layer
inp = Input(shape=INPUT_SHAPE)

# Creating convolutional blocks
x = conv_block(inp, 32)
x = conv_block(x, 32)
x = conv_block(x, 64)
x = conv_block(x, 64)

# Flattening and dense layers
x = Flatten()(x)
x = dense_block(x, 512)
x = dense_block(x, 256)

# Output layer for binary classification
out = Dense(2, activation='softmax')(x)

# Final Model Construction
model = Model(inputs=inp, outputs=out)





# Model compilation
optimizer = Adam(learning_rate=0.001)

model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=[
                  'accuracy',
                  tf.keras.metrics.Precision(name='precision'),
                  tf.keras.metrics.Recall(name='recall')
              ])

model.summary()





# Modle Training and Data Partioning
train_test_split, _, _, _, _ = get_sklearn_components()
to_categorical = get_keras_utilities()

X = np.array(dataset)
Y = to_categorical(np.array(labels))

X_train_val, X_test, y_train_val, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
X_train, X_val, y_train, y_val  = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)





# Implementing Data Augmentation
ImageDataGenerator = get_data_preprocessing_tools()

data_generator = ImageDataGenerator(
    rotation_range    = 15,
    width_shift_range = False,
    height_shift_range= False,
    zoom_range        = False,
    horizontal_flip   = True,
    vertical_flip     = True
)
augmented_data = data_generator.flow(X_train, y_train, batch_size = 64)






# Configuring Training Callbacks
l2, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard = get_training_components()

model_checkpoint = ModelCheckpoint(
    filepath = '/Models/best_CNN_for_Malaria_classifier.keras',
    monitor  = 'accuracy',
    save_best_only = True,
    verbose  = 1,
    mode     = 'max'
)

early_stopping = EarlyStopping(
    monitor  = 'val_loss',
    patience = 5,
    verbose  = 1,
    mode     = 'min'
)

reduce_lr = ReduceLROnPlateau(
    monitor  = 'val_loss',
    factor   = 0.1,
    patience = 4,
    min_lr   = 0.0001,
    verbose  = 1
)

callbacks_list = [model_checkpoint, early_stopping, reduce_lr]




# Executing Model Training
results = model.fit(
    augmented_data,
    verbose = 1,
    epochs  = 15,
    validation_data = (X_val, y_val),
    steps_per_epoch = len(X_train) // 64,
    callbacks = callbacks_list
)




# Model Evaluation
_ , classification_report, confusion_matrix, roc_curve, auc = get_sklearn_components()




# Evalution Metrics
scores = model.evaluate(np.array(X_test), np.array(y_test), verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
print(f"Test Loss: {scores[0]:.5f}")

predictions = model.predict(np.array(X_test))
predicted_classes = np.argmax(predictions, axis=1)

y_test_single_column = np.argmax(y_test, axis=1)

print(classification_report(y_test_single_column, predicted_classes))

average_train_loss = np.mean(results.history['loss'])

final_train_loss = results.history['loss'][-1]

print(f"Average Training Loss: {average_train_loss:.5f}")
print(f"Final Training Loss: {final_train_loss:.5f}")


def plot_training_history(results):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('CNN Performance', fontsize=12)
    fig.subplots_adjust(top=0.85, wspace=0.3)

    max_epoch = len(results.history['accuracy'])
    epoch_list = list(range(1, max_epoch + 1))

    ax1.plot(epoch_list, results.history['accuracy'], label='Train Accuracy')
    ax1.plot(epoch_list, results.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_xticks(np.arange(1, max_epoch + 1, 5))
    ax1.set_ylabel('Accuracy Value')
    ax1.set_xlabel('Epoch')
    ax1.set_title('Accuracy')
    ax1.legend(loc="best")

    ax2.plot(epoch_list, results.history['loss'], label='Train Loss')
    ax2.plot(epoch_list, results.history['val_loss'], label='Validation Loss')
    ax2.set_xticks(np.arange(1, max_epoch + 1, 5))
    ax2.set_ylabel('Loss Value')
    ax2.set_xlabel('Epoch')
    ax2.set_title('Loss')
    ax2.legend(loc="best")

    plt.show()

plot_training_history(results)

conf_matrix = confusion_matrix(y_test_single_column, predicted_classes)
print(conf_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.xticks(ticks=[0.5, 1.5], labels=['Parasitized', 'Uninfected'])
plt.yticks(ticks=[0.5, 1.5], labels=['Parasitized', 'Uninfected'])
plt.show()

y_pred_probs = model.predict(X_test)[:, 1]

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test_single_column, y_pred_probs)

roc_auc = auc(false_positive_rate, true_positive_rate)
print(f'AUC Score: {roc_auc:.2f}')

plt.figure(figsize=(8, 6))
plt.plot(false_positive_rate, true_positive_rate, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()





# Visual Inspection of Model Predictions
class_mapping = {0: "Infected", 1: "Uninfected"}
num_samples = 5

save_directory = '/Images/'

plt.figure(figsize=(15, 3))

for i in range(num_samples):
    sample_index = np.random.choice(len(X_test))
    sample_image = X_test[sample_index]
    predictions = model.predict(np.expand_dims(sample_image, axis=0))
    predicted_class_idx = np.argmax(predictions, axis=1)[0]

    fig = plt.figure(figsize=(64/plt.rcParams['figure.dpi'], 64/plt.rcParams['figure.dpi']), dpi=plt.rcParams['figure.dpi'])
    plt.imshow(sample_image)
    plt.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    sample_image_bgr = cv2.cvtColor(sample_image, cv2.COLOR_RGB2BGR)
    sample_image_uint8 = (sample_image_bgr * 255).astype(np.uint8)

    cv2.imwrite(f"{save_directory}predicted_sample{i+1}.png", sample_image_uint8)
    plt.close(fig)

    ax = plt.subplot(1, num_samples, i + 1)
    plt.imshow(sample_image)
    plt.axis("off")

    actual_class = np.argmax(y_test[sample_index])
    ax.set_title(f"Actual: {class_mapping[actual_class]}\nPredicted: {class_mapping[predicted_class_idx]}")

plt.tight_layout()
plt.show()



# Accuracy - 96.19%


