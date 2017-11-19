"""Module for evaluating recommenders"""

class PrecisionRecall():
    """Class to calculate precision and recall"""

    def compute_precision_recall(self, no_of_items_to_recommend_list, eval_items):
        """calculate the precision and recall measures"""
        #For each distinct no_of_items:
        #    1. For each distinct user, calculate precision and recall.
        #    2. Calculate average precision and recall.
        results = {}

        users_test_sample = eval_items.keys()
        num_users_sample = len(users_test_sample)
        results['no_of_items_to_recommend'] = dict()
        for no_of_items_to_recommend in no_of_items_to_recommend_list:
            results['no_of_items_to_recommend'][no_of_items_to_recommend] = dict()

            sum_precision = 0
            sum_recall = 0
            sum_f1_score = 0
            avg_precision = 0
            avg_recall = 0

            for user_id in users_test_sample:
                items_interacted = eval_items[user_id]['items_interacted']
                items_recommended = eval_items[user_id]['items_recommended'][0:no_of_items_to_recommend]

                hitset = set(items_interacted).intersection(set(items_recommended))
                testset = items_interacted
                no_of_items_interacted = len(testset)
                # if len(hitset) > 0:
                #     print("User ID : ", user_id)
                #     print("Items Interacted : ", set(items_interacted))
                #     print("Items Recommended: ", set(items_recommended))
                #     print("Hitset : ", hitset)
                #     input()

                #precision is the proportion of recommendations that are good recommendations
                precision = float(len(hitset))/no_of_items_to_recommend
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

            results['no_of_items_to_recommend'][no_of_items_to_recommend]['avg_precision'] = round(avg_precision, 4)
            results['no_of_items_to_recommend'][no_of_items_to_recommend]['avg_recall'] = round(avg_recall, 4)
            results['no_of_items_to_recommend'][no_of_items_to_recommend]['avg_f1_score'] = round(avg_f1_score, 4)

        return results
