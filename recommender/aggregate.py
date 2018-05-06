"""Module to aggregate scores of an ensemble of algorithms"""
from pprint import pprint

class Aggregator:
    """Aggregate scores from different algorithms"""
    def __init__(self, data_frame, results_file_name=None):
        self.data_frame = data_frame
        self.results_file_name = results_file_name
        self.df_column_names = set(self.data_frame.columns.values)

    #private methods
    def __is_valid_column_names(self, columns_weights_dict):
        """private function,
        return if columns specified in columns_weights_dict are present in dataframe"""
        for col_name in columns_weights_dict:
            if col_name not in self.df_column_names:
                print("{} doesnt exists in aggregator dataframe".format(col_name))
                return False
        return True

    def __is_valid_weights(self, columns_weights_dict):
        """private function,
        return if weights specified in columns_weights_dict add to 1"""
        try:
            total_weight = round(sum(columns_weights_dict.values()), 2)
            if total_weight != 1.00:
                pprint(columns_weights_dict)
                print("Total Weight of columns = {} do not add to 1".format(total_weight))
                exit(-1)
                return False
            return True
        except TypeError:
            print("Invalid weights specified : ", columns_weights_dict.values())
            return False

    #public methods
    def weighted_avg(self, columns_weights_dict):
        """compute weighted average of columns using column weights provided"""
        aggregation_results = None
        if self.__is_valid_column_names(columns_weights_dict) and \
           self.__is_valid_weights(columns_weights_dict):
            self.data_frame['weighted_avg'] = 0.0
            for index, row in self.data_frame.iterrows():
                total = 0.0
                no_of_cols = 0
                for col_name in columns_weights_dict:
                    total += row[col_name]*columns_weights_dict[col_name]
                    no_of_cols += 1
                if no_of_cols == 0:#To avoid divide by zero
                    no_of_cols = 1
                self.data_frame.loc[index, 'weighted_avg'] = total/no_of_cols
            #print(self.data_frame.head())
            aggregation_results = self.data_frame
            aggregation_results.sort_values('weighted_avg',
                                            ascending=False, inplace=True)
            #aggregation_results['rank'] = range(1, len(aggregation_results)+1)

            if self.results_file_name:
                aggregation_results.to_csv(self.results_file_name)
                print("{:50}    {}".format("Aggregation results are found in : ",
                                           self.results_file_name))
            #print(aggregation_results.head())
            return aggregation_results
        else:
            print("Failed to compute weighted average")
            return aggregation_results
