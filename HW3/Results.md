Data Preview:
   duration protocol_type  service  src_bytes  dst_bytes  flag  hot  num_failed_logins  logged_in  num_compromised  attack
0      0.0         tcp      http      0.0      0.0      SF   0.0                 0.0       0.0              0.0      normal
1      0.0         tcp      http      0.0      0.0      SF   0.0                 0.0       0.0              0.0      normal
2      0.0         tcp      http      0.0      0.0      SF   0.0                 0.0       0.0              0.0      normal
3      0.0         tcp      http      0.0      0.0      SF   0.0                 0.0       0.0              0.0      normal
4      0.0         tcp      http      0.0      0.0      SF   0.0                 0.0       0.0              0.0      normal

Training the model...
Epoch 1/10
4800/4800 [==============================] - 8s 2ms/step - loss: 0.1073 - accuracy: 0.9745 - val_loss: 0.0563 - val_accuracy: 0.9840
Epoch 2/10
4800/4800 [==============================] - 8s 2ms/step - loss: 0.0421 - accuracy: 0.9877 - val_loss: 0.0501 - val_accuracy: 0.9865
Epoch 3/10
4800/4800 [==============================] - 8s 2ms/step - loss: 0.0373 - accuracy: 0.9885 - val_loss: 0.0459 - val_accuracy: 0.9881
...
Epoch 10/10
4800/4800 [==============================] - 8s 2ms/step - loss: 0.0173 - accuracy: 0.9945 - val_loss: 0.0323 - val_accuracy: 0.9910

Evaluation Metrics:
Accuracy: 0.9910
Recall: 0.9875
F1-Score: 0.9932
AUC-ROC: 0.9954
