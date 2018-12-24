from sklearn.preprocessing import MinMaxScaler
import numpy

scaler = MinMaxScaler()
arr = numpy.array([[5.0], [10.0], [800.0]])
rescaled_arr = scaler.fit_transform(arr)
print(rescaled_arr)

# notice how outliers can affect the result badly...
# also, fit_transform needs floats, not ints!

