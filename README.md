# Time Series Forecasting Web Service

This project aims to develop a predictive model for forecasting a certain target variable based on time-series data. The project includes data preprocessing, model training using machine learning and deep learning techniques, and deploying the trained model as a REST API service.

## Technologies Used
- Python 
- TensorFlow
- Keras
- Scikit-learn
- Flask
- Docker

## Data Preprocessing
- Missing values were filled using the mean of each respective feature.
- Features were transformed and engineered to improve model performance.
- RobustScaler was used to normalize the data.
- Feature selection was performed using mutual information regression.

## Model Training
Two types of models were trained:
- **Random Forest Regression:** Utilized from Scikit-learn for traditional machine learning approach.
- **Long Short-Term Memory (LSTM) Neural Network:** Developed using TensorFlow and Keras for deep learning approach.

## Evaluation Metrics
The performance of the trained models was evaluated using the following metrics:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Explained Variance Score (EVS)

## REST API Deployment
The trained LSTM model was deployed as a REST API service using Flask. The API endpoint `/predict` accepts JSON data and returns the predicted value for the target variable.

## Files Description
- `model_training.ipynb`: Jupyter Notebook containing the process of training both Random Forest and LSTM models.
- `main.py`: Python script implementing the Flask application for the REST API service.
- `modelLSTM.h5`: Trained LSTM model saved in HDF5 format.
- `random_forest_model.pkl`: Trained Random Forest model saved using pickle serialization.
- `scaler`: Serialized RobustScaler object used for data normalization.

## Setup

1. **Clone the Repository**: Start by cloning this repository to your local machine.

    ```bash
    git clone <repository_url>
    ```

2. **Install Dependencies**: Navigate to the project directory and install the required Python packages by running:

    ```bash
    pip install -r requirements.txt
    ```

3. **Prepare Data**: Ensure you have the right dataset `xxx.csv` in your project directory. You may need to preprocess the data as specified in the task requirements.

4. **Train Models**: Execute the Jupyter notebook `model_training.ipynb` to train the predictive models and save them.

5. **Build Docker Image**: Build the Docker image using the provided `Dockerfile`.

    ```bash
    docker build -t time-series-forecasting .
    ```

6. **Run Docker Container**: Run the Docker container once the image is built.

    ```bash
    docker run -p 5000:5000 time-series-forecasting
    ```

## Usage

### Making Predictions

You can make predictions by sending a POST request to the `/predict` endpoint with JSON data containing the input features. Here's an example using `curl`:

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"time": "2024-03-27T00:00:00", "mean T": 20.5, "min T": 18.2, "max T": 24.8, "mean rel. hum.": 70, "min rel. hum.": 60, "max rel. hum.": 80, "T": 22.3, "rel. hum.": 75, "precipitation": 0.2, "wind speed": 15.5, "wind direction": 180, "max gust": 25.0, "diffusive energy": 50.0}' \
  http://localhost:5000/predict
```

Replace the JSON data with your input features accordingly.

### Testing with Postman

You can also test the web service using Postman or similar tools by sending POST requests to the `/predict` endpoint with appropriate JSON data.

## Additional Notes

- Ensure that the dataset is preprocessed and features are selected as per the task requirements before training the models.
- The trained models (`modelLSTM.h5` and `random_forest_model.pkl`) should be saved in the project directory.
- The `scaler` object should also be serialized and saved as `scaler` to ensure consistency in feature scaling during predictions.

## Conclusion
This project demonstrates the implementation of machine learning and deep learning techniques for time-series forecasting tasks. By combining traditional and neural network models, along with appropriate preprocessing techniques, accurate predictions can be achieved.

