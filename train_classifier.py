import numpy as np
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
import json
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def main():
    input_file = "embeddings.npz"
    output_model = "impressionism_classifier.keras"
    
    # 1. Load precomputed embeddings
    print(f"Loading embeddings from {input_file}...")
    try:
        data = np.load(input_file)
        X = data["embeddings"]
        labels = data["labels"]
        print(f"Loaded {X.shape[0]} samples with feature dimension {X.shape[1]}")
    except Exception as e:
        print(f"Error loading {input_file}: {e}")
        print("Please run extract_embeddings.py first.")
        return
        
    # 2. Prepare binary targets (1 for Impressionism, 0 for other styles)
    y = (labels == "Impressionism").astype(np.float32)
    
    # Print class distribution
    num_impressionism = np.sum(y == 1)
    num_others = np.sum(y == 0)
    print(f"Class distribution:")
    print(f"  Impressionism: {num_impressionism} ({num_impressionism / len(y) * 100:.1f}%)")
    print(f"  Other Styles:  {num_others} ({num_others / len(y) * 100:.1f}%)")
    
    # 3. Stratified split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {X_train.shape[0]}, Validation size: {X_val.shape[0]}")
    
    # 4. Build lightweight Keras model (Linear Classifier / Logistic Regression)
    print("Building Keras classification model...")
    model = keras.Sequential([
        keras.layers.Input(shape=(512,)),
        keras.layers.Dense(1, activation="sigmoid", kernel_regularizer=keras.regularizers.l2(1e-4))
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall")
        ]
    )
    
    model.summary()
    
    # 5. Train the model with Early Stopping to prevent overfitting
    print("Training model...")
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=40,  # Linear model learns fast and is highly stable
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # 6. Evaluate the model
    print("\nEvaluating model on validation set...")
    val_loss, val_acc, val_precision, val_recall = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Loss:      {val_loss:.4f}")
    print(f"Validation Accuracy:  {val_acc*100:.2f}%")
    print(f"Validation Precision: {val_precision*100:.2f}%")
    print(f"Validation Recall:    {val_recall*100:.2f}%")
    
    # Calculate F1 score
    val_f1 = 0.0
    if val_precision + val_recall > 0:
        val_f1 = float(2 * (val_precision * val_recall) / (val_precision + val_recall))
        print(f"Validation F1-Score:  {val_f1*100:.2f}%")
        
    # Detailed metrics using scikit-learn
    print("\nCalculating detailed classification metrics...")
    y_val_pred_prob = model.predict(X_val, verbose=0)
    y_val_pred = (y_val_pred_prob > 0.5).astype(np.int32)
    
    tn, fp, fn, tp = confusion_matrix(y_val, y_val_pred).ravel()
    roc_auc = float(roc_auc_score(y_val, y_val_pred_prob))
    report = classification_report(y_val, y_val_pred, target_names=["Other Styles", "Impressionism"], output_dict=True)
    
    # Save training history and validation metrics to JSON
    metrics_log = {
        "final_metrics": {
            "validation_loss": float(val_loss),
            "validation_accuracy": float(val_acc),
            "validation_precision": float(val_precision),
            "validation_recall": float(val_recall),
            "validation_f1_score": val_f1,
            "roc_auc": roc_auc,
            "confusion_matrix": {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp)
            },
            "classification_report": report
        },
        "training_history": {
            key: [float(val) for val in values] 
            for key, values in history.history.items()
        }
    }
    
    metrics_file = "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_log, f, indent=4)
    print(f"Metrics and training history saved to {metrics_file}")
    
    # Try to plot and save visualization curves
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay
        
        print("\nGenerating metrics plots...")
        
        # 1. Plot Training History (Loss and Accuracy)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.plot(history.history['loss'], label='Train Loss', marker='o')
        ax1.plot(history.history['val_loss'], label='Val Loss', marker='s')
        ax1.set_title('Training & Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(history.history['accuracy'], label='Train Acc', marker='o')
        ax2.plot(history.history['val_accuracy'], label='Val Acc', marker='s')
        ax2.set_title('Training & Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig("learning_curves.png", dpi=300)
        plt.close()
        print("Saved learning_curves.png")
        
        # 2. Plot Confusion Matrix
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(
            y_val, y_val_pred, 
            display_labels=["Other", "Impressionism"], 
            cmap="Blues", 
            ax=ax
        )
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png", dpi=300)
        plt.close()
        print("Saved confusion_matrix.png")
        
        # 3. Plot ROC Curve
        fig, ax = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_predictions(y_val, y_val_pred_prob, ax=ax)
        ax.plot([0, 1], [0, 1], 'k--', label='Random Guess')
        ax.set_title("ROC Curve")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig("roc_curve.png", dpi=300)
        plt.close()
        print("Saved roc_curve.png")
        
    except ImportError:
        print("\nNote: matplotlib is not installed or couldn't be imported. Plots were not generated.")
        print("To generate plots, install matplotlib by running: pip install matplotlib")
    except Exception as e:
        print(f"\nWarning: Could not generate plots: {e}")

    # 7. Save model to disk
    print(f"\nSaving model to {output_model}...")
    model.save(output_model)
    print("Model saved successfully!")

if __name__ == "__main__":
    main()
