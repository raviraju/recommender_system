"""Module for evaluating recommenders"""

import random

class PrecisionRecall():
    """Class to calculate precision and recall"""
    def __init__(self, train_data, test_data, model, user_id_col, item_id_col):
        """constructor"""
        self.test_data = test_data
        self.train_data = train_data
        self.model = model
        self.user_id_col = user_id_col
        self.item_id_col = item_id_col

        self.users_test_sample = None

    def get_random_sample(self, list_a, percentage):
        """Return random percentage of values from a list"""
        k = int(len(list_a) * percentage)
        random.seed(0)
        indicies = random.sample(range(len(list_a)), k)
        new_list = [list_a[i] for i in indicies]
        return new_list

    def fetch_user_test_sample(self, percentage):
        """return test sample of users"""
        #Find users common between training and test set
        users_set_train_data = set(self.train_data[self.user_id_col].unique())
        users_set_test_data = set(self.test_data[self.user_id_col].unique())
        users_test_and_training = list(users_set_test_data.intersection(users_set_train_data))
        print("No of users common between training and test set : {}".format(len(users_test_and_training)))

        #Take only random user_sample of common users for evaluation
        users_test_sample = self.get_random_sample(users_test_and_training, percentage)
        print("Sample no of common users, used for evaluation : {}".format(len(users_test_sample)))
        return users_test_sample

    def generate_recommendations(self):
        """Generate recommendations for users in the user test sample"""
        items_predicted = dict()
        items_interacted = dict()
        for user_id in self.users_test_sample:
            #Get items for user_id from recommendation model
            #print("Getting recommendations for user:{}".format(user_id))
            recommended_items = self.model.recommend(user_id)
            items_predicted[user_id] = recommended_items

            #Get items for user_id from test_data
            test_data_user = self.test_data[self.test_data[self.user_id_col] == user_id]
            items_interacted[user_id] = test_data_user[self.item_id_col].unique()
        return items_predicted, items_interacted

    def compute_precision_recall(self, no_of_items_to_predict_list, items_predicted, items_interacted):
        """calculate the precision and recall measures"""
        #Create no_of_items_list for precision and recall calculation
        #no_of_items_to_predict_list = [20, 30]#list(range(1, 11))

        #For each distinct cutoff:
        #    1. For each distinct user, calculate precision and recall.
        #    2. Calculate average precision and recall.
        results = {}

        num_users_sample = len(self.users_test_sample)
        results['no_of_items_to_predict'] = dict()
        for no_of_items_to_predict in no_of_items_to_predict_list:
            results['no_of_items_to_predict'][no_of_items_to_predict] = dict()

            sum_precision = 0
            sum_recall = 0
            sum_f1_score = 0
            avg_precision = 0
            avg_recall = 0

            for user_id in self.users_test_sample:
                
                hitset = set(items_interacted[user_id]).intersection(set(items_predicted[user_id][0:no_of_items_to_predict]))
                testset = items_interacted[user_id]
                no_of_items_interacted = len(testset)
                # if len(hitset) > 0:
                #     print("User ID : ", user_id)
                #     print("Items Interacted : ", set(items_interacted[user_id]))
                #     print("Items Recommended: ", set(items_predicted[user_id][0:no_of_items]))
                #     print("Hitset : ", hitset)
                #     input()

                #precision is the proportion of recommendations that are good recommendations
                precision = float(len(hitset))/no_of_items_to_predict
                #recall is the proportion of good recommendations that appear in top recommendations
                recall = float(len(hitset))/no_of_items_interacted
                if (recall+precision) != 0:
                    f1_score = float(2*precision*recall)/(recall+precision)
                else:
                    f1_score = 0.0
                sum_precision += precision
                sum_recall += recall                
                sum_f1_score += f1_score

            avg_precision = sum_precision/float(num_users_sample)
            avg_recall = sum_recall/float(num_users_sample)
            avg_f1_score = sum_f1_score/float(num_users_sample)

            results['no_of_items_to_predict'][no_of_items_to_predict]['avg_precision'] = round(avg_precision, 2)
            results['no_of_items_to_predict'][no_of_items_to_predict]['avg_recall'] = round(avg_recall, 2)
            results['no_of_items_to_predict'][no_of_items_to_predict]['avg_f1_score'] = round(avg_f1_score, 2)

        return results

    def compute_measures(self, no_of_items_to_predict_list, test_users_percentage):
        """compute precision recall using percentage of test users"""
        #Fetch a sample of common users from test and training set
        self.users_test_sample = self.fetch_user_test_sample(test_users_percentage)

        #Generate recommendations for the test sample users
        items_predicted, items_interacted = self.generate_recommendations()

        #Calculate precision and recall at different cutoff values
        return self.compute_precision_recall(no_of_items_to_predict_list, items_predicted, items_interacted)
