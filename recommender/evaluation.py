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
        print("No of users common between training and test set:{}".format(len(users_test_and_training)))

        #Take only random user_sample of common users for evaluation
        users_test_sample = self.get_random_sample(users_test_and_training, percentage)
        print("No of sample users for evaluation:{}".format(len(users_test_sample)))
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

    def compute_precision_recall(self, items_predicted, items_interacted):
        """calculate the precision and recall measures"""
        #Create no_of_items_list for precision and recall calculation
        no_of_items_list = [1, 5, 10]#list(range(1, 11))

        #For each distinct cutoff:
        #    1. For each distinct user, calculate precision and recall.
        #    2. Calculate average precision and recall.

        avg_precision_list = []
        avg_recall_list = []

        num_users_sample = len(self.users_test_sample)
        for no_of_items in no_of_items_list:
            sum_precision = 0
            sum_recall = 0
            avg_precision = 0
            avg_recall = 0

            for user_id in self.users_test_sample:
                print(user_id)
                print("Items Interacted : ", set(items_interacted[user_id]))
                print("Items Recommended: ", set(items_predicted[user_id][0:no_of_items]))

                hitset = set(items_interacted[user_id]).intersection(set(items_predicted[user_id][0:no_of_items]))
                testset = items_interacted[user_id]
                print("Hitset : ", hitset)

                sum_precision += float(len(hitset))/float(no_of_items)
                sum_recall += float(len(hitset))/float(len(testset))

            avg_precision = sum_precision/float(num_users_sample)
            avg_recall = sum_recall/float(num_users_sample)

            avg_precision_list.append(avg_precision)
            avg_recall_list.append(avg_recall)

        return (avg_precision_list, avg_recall_list)

    def compute_measures(self, test_users_percentage):
        """compute precision recall using percentage of test users"""
        #Fetch a sample of common users from test and training set
        self.users_test_sample = self.fetch_user_test_sample(test_users_percentage)

        #Generate recommendations for the test sample users
        items_predicted, items_interacted = self.generate_recommendations()

        #Calculate precision and recall at different cutoff values
        return self.compute_precision_recall(items_predicted, items_interacted)
