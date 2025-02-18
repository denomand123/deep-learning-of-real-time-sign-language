import pickle
import matplotlib.pyplot as plt

# Load the training history
with open('history.pkl', 'rb') as file:
    history = pickle.load(file)

# Plot accuracy
plt.figure(figsize=(10, 6))
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('model_accuracy.png')
plt.show()

# Plot loss
plt.figure(figsize=(10, 6))
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig('model_loss.png')
plt.show()
