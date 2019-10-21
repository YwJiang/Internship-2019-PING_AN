import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Multiply, Embedding, Flatten, LSTM, Dropout, Bidirectional
from tensorflow.keras.layers import concatenate
from tensorflow.python.keras import backend as K

from text_preprocessing import text_df_to_array_pad
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
import time


def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision. Computes the precision, a
    metric for multi-label classification of how many selected items are
    relevant.
    """
#     print("true_positives: ", K.clip(y_true * y_pred, 0, 1))
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     print("true_positives: ", true_positives)
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
#     print("precision: ", precision)
    return precision

def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall. Computes the recall, a metric
    for multi-label classification of how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
#     print("recall: ", recall)
    return recall

def f1_score(y_true, y_pred):
    """Computes the F1 Score
    Only computes a batch-wise average of recall. Computes the recall, a metric
    for multi-label classification of how many relevant items are selected.
    """
#     print(y_true, y_pred)
#     print(y_true.shape, y_pred.shape)
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return (2 * p * r) / (p + r + K.epsilon())


def split_data(X, y, test_size=0.33):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=38)
    X_validation, X_test, y_validation, y_test = train_test_split(X_test, y_test, random_state=38)
    return X_train, X_validation, X_test, y_train, y_validation, y_test

def build_model1(padding_len=256, dict_len=10, output_dim=70,
                 con_op=100, kernel_size=3, output_cells=2):
    inputs = Input(shape=(padding_len,),name="input_layer")
    emb = Embedding(dict_len, output_dim, input_length=None, name="embedding_layer")(inputs)

    # Attention part starts here
    attention_probs = Dense(output_dim, activation="softmax", name="attention_vec")(emb)
    attention_mul = Multiply()([emb, attention_probs])
    # Attention part finishes here

    con = Conv1D(con_op, kernel_size, padding="same", strides=1, activation="relu",
                 name="convolution1")(attention_mul)

    flat = Flatten()(con)
    output = Dense(output_cells, activation="sigmoid")(flat)
    model = Model(inputs=inputs, outputs=output)
    return model

def build_model1_3(padding_len=256, dict_len=10, output_dim=70,
                 con_op=100, kernel_size=3, output_cells=2):
    inputs = Input(shape=(padding_len,), name="input_layer")
    emb = Embedding(dict_len, output_dim, input_length=None, name="embedding_layer")(inputs)

    # Attention part starts here
    attention_probs = Dense(output_dim, activation="softmax", name="attention_vec")(emb)
    attention_mul = Multiply()([emb, attention_probs])
    # Attention part finishes here

    con = Conv1D(con_op, kernel_size, padding="same", strides=1, activation="relu",
                 name="convolution1")(attention_mul)

    # LSTM
    lstm = Bidirectional(LSTM(140, input_dim=64, input_length=10, return_sequences=True))(con)

    flat = Flatten()(lstm)
    drop = Dropout(0.5)(flat)
    output = Dense(output_cells, activation="sigmoid")(drop)
    model = Model(inputs=inputs, outputs=output)
    return model


def compile_and_train(dict_len, padding_len, X_train, X_test, y_train, y_test):
    model = build_model1_3(padding_len=padding_len, dict_len=dict_len, output_cells=y_train.shape[1])
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=[f1_score])
    early_stopping = EarlyStopping(monitor='f1_score', min_delta=0.001, patience=30)
    history = model.fit(X_train, y_train, epochs=25, batch_size=50,
                        validation_data=(X_test, y_test), verbose=1,
                        callbacks=[early_stopping])
    return model.summary, history, model

def create_model(label_num=3, padding_len=500, dict_len=10,
                 output_dim=100, con_op=100,
                 kernel_size=3, optimizer='rmsprop'):
    model = build_model1(padding_len=padding_len, dict_len=dict_len,
                         output_dim=output_dim, con_op=con_op,
                         kernel_size=kernel_size, output_cells=label_num)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=['accuracy'])
    return model

def grid_search(padding_len, dict_len, X, y):
    model = KerasClassifier(build_fn=create_model)
    optimizers = ['rmsprop', 'adam']
    con_ops = np.array([20, 40, 60, 100])
    epochs = np.array([30, 50, 100])
    batches = np.array([2, 5, 10, 20])

    # the following params are not searched
    dict_len = np.array([dict_len])
    padding_len = np.array([padding_len])
    label_num = np.array([y.shape[1]])

    param_grid = dict(optimizer=optimizers,np_epoch=epochs,
                      batch_size=batches, con_op=con_ops,
                      dict_len=dict_len, padding_len=padding_len,
                      label_num=label_num)
    grid = GridSearchCV(estimator=model, param_grid=param_grid)
    grid_result = grid.fit(X, y)

    return grid_result.best_score_, grid_result.best_params_, grid_result.best_estimator_


def plot_fmeasure(history):
    print("plot starts")
    acc = history.history['f1_score']
    val_acc = history.history['val_f1_score']

    epochs = range(1, len(acc)+1)

    # "bo" is for "blue dot"
    plt.plot(epochs, acc, 'bo', label="Training acc")
    # b is for "solid blue line"
    plt.plot(epochs, val_acc, "b", label="Validation acc")
    plt.title('Training and validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('ACC')
    plt.lengend()

    plt.show()
    print("plot finished!")


def plot_acc(history):
    print("plot starts")
    acc = history.history['acc']
    val_acc = history.history['val_acc']

    epochs = range(1, len(acc) + 1)

    # "bo" is for "blue dot"
    plt.plot(epochs, acc, 'bo', label="Training acc")
    # b is for "solid blue line"
    plt.plot(epochs, val_acc, "b", label="Validation acc")
    plt.title('Training and validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('ACC')
    plt.lengend()

    plt.show()
    print("plot finished!")


def save_model(model, path="./model/model1.h5"):
    model.save(path)


def load_predict(X_test, path="./model/model1.h5"):
    model = load_model(path, custom_objects={'f1_score': f1_score}, compile=False)
    y_predict = model.predict(X_test)

    return model, y_predict


if __name__ == "__main__":
    from sklearn.metrics import accuracy_score
    path = "../data/credit_card_labels_factreason.xlsx"
    pd_all = pd.read_excel(path)
    pd_all = pd_all.fillna(0)
    X_array, dict_len, padding_len = text_df_to_array_pad(pd_all, 'fact_reason', pad_len=500)
    y_label = pd_all.iloc[:, 0:3]

    # 普通 train and test
    X_train, X_validation, X_test, y_train, y_validation, y_test=split_data(X_array, y_label)

    start = time.time()
    early_stopping = EarlyStopping(monitor='f1_score', min_delta=0, patience=30)

    _, hist, model = compile_and_train(dict_len, padding_len, X_train, X_validation, y_train, y_validation)

    save_model(model, "../model/model2.h5")

    model, y_predict = load_predict(X_test, "../model/model2.h5")

    print(y_predict)
    print(type(y_predict))

    y_pred = K.round(y_predict)
    y_pred = tf.Session().run(y_pred)

    print(accuracy_score(y_test, y_pred, normalize=True, sample_weight=None))

    end = time.time()
    print(end-start)