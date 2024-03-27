import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import tensorflow as tf


def create_dataset(dataset, look_back=1):
    data_x, data_y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        data_x.append(a)
        data_y.append(dataset[i + look_back, 0])
    return np.array(data_x), np.array(data_y)


model = tf.keras.models.load_model('modelLSTM.h5')
scaler = pickle.load(open('scaler', 'rb'))
regressor = pickle.load(open('random_forest_model.pkl', 'rb'))


app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame(data)

        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values(by='time')
        df[['precipitation', 'wind speed', 'wind direction', 'max gust', 'diffusive energy']] = df[
            ['precipitation', 'wind speed', 'wind direction', 'max gust', 'diffusive energy']].fillna(value=df.mean(numeric_only=True))
        print(df)

        X_test_global = df.drop(['global energy', 'time'], axis=1)
        y_pred = regressor.predict(X_test_global)
        print("y_pred:")
        print(y_pred)

        missing_indexes = df[df['global energy'].isnull()].index

        df.loc[missing_indexes, 'global energy'] = y_pred[:len(missing_indexes)]

        print("missing:")
        print(missing_indexes)
        print("df missing")
        print(df)

        df['combined_feature'] = df['precipitation'] + df['max gust']
        new_df = df.drop(['time','T', 'precipitation', 'max gust', 'min rel. hum.', 'mean rel. hum.', 'max rel. hum.', 'max T', 'mean T', 'min T', 'wind speed', 'wind direction'], axis=1)
        print(new_df)

        try:
            window_size = 24 * 6
            train_data_scaled = scaler.fit_transform(new_df)
            X_test, y_test = create_dataset(train_data_scaled, window_size)
            print("X_test:")
            print(X_test)
            print("y_test:")
            print(y_test)
            X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
            print("X_test:")
            print(X_test)

            predictions = model.predict(X_test)
            print("predictions:")
            print(predictions)

            response = {'prediction': float(predictions[0])}

            return jsonify(response)

        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            return jsonify({'error': error_message})

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        return jsonify({'error': error_message})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
