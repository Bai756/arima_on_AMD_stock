import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def difference(data, order):
    diff = data
    for i in range(order):
        diff[i] = diff[i] - diff[i - 1]
    return diff

def autoregression(data, lags, coefficients):
    total = 0
    reverse_data = data[::-1]
    for i in range(lags):
        total += coefficients[i] * reverse_data[i]
    return total

def moving_average(errors, lags, coefficients):
    total = 0
    reverse_errors = errors[::-1]
    for i in range(lags):
        total += coefficients[i] * reverse_errors[i]
    return total

def arima_predict(data, ar_coeff, ma_coeff, d, errors):
    stationary_data = difference(data, d)
    
    ar = autoregression(stationary_data, len(ar_coeff), ar_coeff)
    ma = moving_average(errors, len(ma_coeff), ma_coeff)
    
    prediction =  (ar + ma) / 2
    return prediction


def autoreg(data, window, iterations, learning_rate):
    n = len(data)
    q = window
    coefficients = [1] * q 

    predictions = []

    for iteration in range(iterations):
        total_error = 0

        for t in range(q, n):
            # Generate prediction
            predict = 0
            for lag in range(q):
                predict += coefficients[lag] * data[t - lag - 1]
            predictions.append(predict)

            # Calculate error
            error = data[t] - predict
            total_error += error ** 2

            # Gradient descent
            for lag in range(q):
                grad = learning_rate * error * data[t - lag - 1]
                grad = np.clip(grad, -5, 5)
                coefficients[lag] += grad

        if iteration % 100 == 0:
            print(f'Iteration: {iteration}, Error: {total_error}')

    return coefficients

def moving_average_model(data, order, iterations, learning_rate):
    n = len(data)
    q = order
    coefficients = [1] * q
    mean = np.mean(data)
    errors = [0] * q 
    
    for iteration in range(iterations):
        predictions = []
        total_error = 0

        for t in range(n):
            # Generate prediction
            predict = mean
            for lag in range(q):
                if t - lag - 1 >= 0: 
                    predict += coefficients[lag] * errors[-(lag + 1)]
            predictions.append(predict)
            
            # Error
            if t < n:
                error = data[t] - predict
                total_error += error ** 2

                # Gradient descent
                for lag in range(q):
                    if t - lag - 1 >= 0:
                        grad = learning_rate * error * errors[-(lag + 1)]
                        grad = np.clip(grad, -5, 5)
                        coefficients[lag] += grad
            
                errors.append(error)
                errors.pop(0)
        if iteration % 100 == 0:
            print(f'Iteration: {iteration}, Error: {total_error}')
    
    return coefficients, errors


learning_rate = 0.00002
num_iterations = 1001
order = 20
difference_order = 1
num_of_predictions = 30 # subtract by order

data = pd.read_csv('AMD-stock.csv')
data = data[['Date', 'Close/Last']]
data['Date'] = pd.to_datetime(data['Date'])
data = data.iloc[::-1]
data['Close/Last'] = data['Close/Last'].str.replace('$', '')
data['Close/Last'] = data['Close/Last'].astype(float)

normalized_data = data.copy()
normalized_data['Close/Last'] = (data['Close/Last'] - np.mean(data['Close/Last'])) / np.std(data['Close/Last'])
normalized_data = normalized_data.set_index('Date')
normalized_data = normalized_data.values

train = normalized_data[:-num_of_predictions]
test = normalized_data[-num_of_predictions:]

ma_coeff, errors = moving_average_model(train, order, num_iterations, learning_rate)
ar_coeff = autoreg(train, order, num_iterations, learning_rate)

predictions = []
for t in range(len(test) - order):
    prediction = arima_predict(test[t:t + order], ar_coeff, ma_coeff, difference_order, errors)
    predictions.append(prediction.item())

    actual = test[t + order]
    errors.append(actual - prediction)
    errors.pop(0)

full_predictions = []
for t in range(len(normalized_data) - order):
    prediction = arima_predict(normalized_data[t:t + order], ar_coeff, ma_coeff, difference_order, errors)
    full_predictions.append(prediction.item())

    actual = normalized_data[t + order]
    errors.append(actual - prediction)
    errors.pop(0)

# Unnormalize
predictions = np.array(predictions) * np.std(data['Close/Last']) + np.mean(data['Close/Last'])
full_predictions = np.array(full_predictions) * np.std(data['Close/Last']) + np.mean(data['Close/Last'])
full_predictions = full_predictions[order:] # Remove the first predictions because error is wrong there

rmse = np.sqrt(np.mean((data['Close/Last'][-num_of_predictions + order:] - predictions) ** 2))
full_rsme = np.sqrt(np.mean((data['Close/Last'][order + order:] - full_predictions) ** 2))
print(f'Full RMSE: {full_rsme}')
print(f'10 Day RMSE: {rmse}')

plt.plot(data['Date'][-len(full_predictions):], full_predictions, color='red', label='Predictions')
plt.plot(data['Date'][-len(full_predictions):], data['Close/Last'][-len(full_predictions):], color='blue', label='Actual')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.title('Full Predictions vs Actual')
plt.show()

plt.plot(data['Date'][-len(predictions):], predictions, color='red', label='Predictions')
plt.plot(data['Date'][-len(predictions):], data['Close/Last'][-len(predictions):], color='blue', label='Actual')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.title('10 Day Predictions vs Actual')
plt.show()
