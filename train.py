from sklearn.metrics import mean_absolute_percentage_error
import data
import model

data_df = data.read_data('wine_quality.csv')
inputs, targets = data.generate_dataset(data_df)

scaled_inputs, x_scaler = data.scale(inputs)
scaled_targets, y_scaler = data.scale(targets.to_numpy().reshape((-1, 1)))

x_train, x_test, y_train, y_test = data.train_test_split(scaled_inputs, scaled_targets, test_size=0.1, random_state=42)

clf = model.build_model(input_shape=x_train.shape[1:], hidden_units=[128, 128, 128])
model.train(clf, 'wine_quality_clf', x_train, y_train, batch_size=16, epochs=50)

clf = model.load_model('weights/wine_quality_clf_callback.h5')
y_predictions = model.predict(clf, x_test)

y_predictions = data.inverse_scaling(y_predictions, y_scaler)
y_test = data.inverse_scaling(y_test, y_scaler)

error = mean_absolute_percentage_error(y_test, y_predictions)
print('Mean Absolute Percentage ERROR =', error, '\n')

print('\nPrediction | Actual')
for pred, y_actual in zip(y_predictions, y_test):
    print(pred[0], '|', y_actual[0])
