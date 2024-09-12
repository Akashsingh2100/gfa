import model
from datasets import Dataset
import yaml
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

# Configure GPU memory growth (if needed)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

if __name__ == '__main__':
    with open("E:/akash singh/GFA-CNN-new/GFA-CNN-master/GFA-CNN-master/config.yaml", "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)  # Use safe_load for security

    epoch = cfg['training']['epoch']
    batch = cfg['training']['batch']

    attack_dir = r"E:\akash singh\GFA-CNN-new\GFA-CNN-master\GFA-CNN-master\dataset_new\msumsfd_gfacnn\attack"
    real_dir = r"E:\akash singh\GFA-CNN-new\GFA-CNN-master\GFA-CNN-master\dataset_new\msumsfd_gfacnn\real"
    dataset = Dataset('MSU-MSFD', batch_size=batch, attack_dir=attack_dir, real_dir=real_dir)
    GFA_CNN = model.get_model()

    checkpoint_filepath = cfg['training']['checkpoint']
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='loss',
        mode='min',
        save_best_only=True)

    print('Preprocessing data ...')

    print('Training ...')
    j = 1
    for e in tqdm(range(epoch), desc='Epochs'):
        print(f"[epoch number: {j}]")
        dataset.shuffle_dataset()
        i = 1
        for batch_img_4_truth, batch_label_truth, batch_img_random_1, batch_img_random_2, batch_label_random in tqdm(
                dataset.generate_minibatch(), desc='Batches', total=len(dataset.list_file_path_truth) // batch):
            GFA_CNN.fit(
                {"as_input": batch_img_4_truth, "lpc_input_1": batch_img_random_1, "lpc_input_2": batch_img_random_2},
                {"as_output": batch_label_truth, 'lpc': batch_label_random},
                epochs=1,
                batch_size=16,
                callbacks=[model_checkpoint_callback],
                verbose=1
            )
            print(f"counting iterations {i}")
            i += 1
        j += 1

    print("Saving the model...")
    model_save_path = cfg['training']['model_save_path']
    GFA_CNN.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    # Load the testing dataset
    attack_dir_test = r"E:\akash singh\GFA-CNN-new\GFA-CNN-master\GFA-CNN-master\dataset_new\msumsfd_gfacnn_test\attack"
    real_dir_test = r"E:\akash singh\GFA-CNN-new\GFA-CNN-master\GFA-CNN-master\dataset_new\msumsfd_gfacnn_test\real"
    test_dataset = Dataset('MSU-MSFD', batch_size=batch, attack_dir=attack_dir_test, real_dir=real_dir_test)

    y_true = []
    y_pred = []

    for batch_img_4_truth, batch_label_truth, batch_img_random_1, batch_img_random_2 in test_dataset.generate_minibatch():
        predictions = GFA_CNN.predict({
            "as_input": batch_img_4_truth,
            "lpc_input_1": batch_img_random_1,
            "lpc_input_2": batch_img_random_2
        })

        y_true.extend(batch_label_truth)
        y_pred.extend(predictions[0])  # Adjust according to your output names

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred = y_pred[:, 0]  # Assuming first column represents positive class probability

    def calculate_eer(y_true, y_pred):
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        fnr = 1 - tpr
        eer_threshold = thresholds[np.nanargmin(np.abs(fnr - fpr))]
        eer = fpr[np.nanargmin(np.abs(fnr - fpr))]
        return eer

    def calculate_hter(eer, y_true, y_pred):
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        fnr = 1 - tpr
        threshold = np.interp(eer, fpr, tpr)
        hter = (fpr[np.argmin(np.abs(threshold - tpr))] + fnr[np.argmin(np.abs(threshold - fnr))]) / 2
        return hter

    eer = calculate_eer(y_true, y_pred)
    print(f"Equal Error Rate (EER): {eer:.4f}")

    hter = calculate_hter(eer, y_true, y_pred)
    print(f"Half Total Error Rate (HTER): {hter:.4f}")

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()
