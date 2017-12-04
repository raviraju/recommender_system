import numpy as np

from sklearn.preprocessing import MinMaxScaler

debug = True

class MinMaxScaling:

    def __init__(self):
        self.min_max_scaler = None

    # Passing min and max value is optional
    def scale(self, numpy_array, min_scale=0, max_scale=1):
        try:
            input_array = numpy_array
            input_array = input_array.reshape((len(input_array), 1))
            min_val = np.amin(input_array)
            max_val = np.amax(input_array)
            scaled_ndarray = None
            if min_val == max_val and max_val > 0:
                no_of_values = input_array.shape[0]
                scaled_ndarray = np.ones(no_of_values)
            elif min_val == max_val and max_val == 0:
                no_of_values = input_array.shape[0]
                scaled_ndarray = np.zeros(no_of_values)
            else:
                self.min_max_scaler = MinMaxScaler(feature_range=(min_scale, max_scale))
                self.min_max_scaler = self.min_max_scaler.fit(input_array)
                scaled_ndarray = self.min_max_scaler.transform(input_array)
            scaled_values = scaled_ndarray.flatten()
            return scaled_values.tolist()
        except Exception as e:
            if debug:
                print("Error in Min Max Scaling")
                print(e)
            return None


    def descale(self, scaled_ndarray):
        try:
            if self.min_max_scaler is not None:
                inversed = self.min_max_scaler.inverse_transform(scaled_ndarray)
                descaled_ndarray = np.round(inversed, decimals=2)
                if descaled_ndarray is not None:
                    descaled_ndarray = descaled_ndarray.flatten()
                return descaled_ndarray
            else:
                return None
        except Exception as e:
            if debug:
                print("Error in Min Max Descaling")
                print(e)
            return None
    
def main():
    # Unit Testing
    M = MinMaxScaling()

    input_array = np.array([4.2,5,100,-270.1])
    print("Input Array : " + str(input_array))
    
    scaled_ndarray = M.scale(input_array,10,20)
    
    if scaled_ndarray is not None:
        print("Min Max Scaled Array : " + str(scaled_ndarray.flatten()))
    
    descaled_ndarray = M.descale(scaled_ndarray)
    print("Min Max Descaled Array : " + str(descaled_ndarray))
    
    
if __name__ == '__main__':
    main()