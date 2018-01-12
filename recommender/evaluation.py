"""Module for evaluating recommenders"""
import math


class PrecisionRecall():
    """Class to calculate precision and recall"""

    def compute_precision_recall(self, no_of_items_to_recommend_list, eval_items, all_items=None):
        """calculate the precision and recall measures"""
        # For each distinct no_of_items:
        #    1. For each distinct user, calculate precision and recall.
        #    2. Calculate average precision and recall.
        results = {}

        users_test_sample = eval_items.keys()
        num_users_sample = len(users_test_sample)
        results['no_of_items_to_recommend'] = dict()
        for no_of_items_to_recommend in no_of_items_to_recommend_list:
            results['no_of_items_to_recommend'][
                no_of_items_to_recommend] = dict()

            sum_precision = 0
            sum_recall = 0
            sum_f1_score = 0
            sum_mcc_score = 0

            for user_id in users_test_sample:
                items_interacted = eval_items[user_id]['items_interacted']
                items_recommended = eval_items[user_id][
                    'items_recommended'][0:no_of_items_to_recommend]
                items_assumed_to_be_interacted = eval_items[
                    user_id]['assume_interacted_items']
                #hitset = set(items_interacted).intersection(set(items_recommended))
                #no_of_items_interacted = len(items_interacted)
                # if len(hitset) > 0:
                #     print("User ID : ", user_id)
                #     print("Items Interacted : ", set(items_interacted))
                #     print("Items Recommended: ", set(items_recommended))
                #     print("Hitset : ", hitset)
                #     input()
                items_interacted_set = set(items_interacted)
                items_recommended_set = set(items_recommended)
                items_not_interacted_set = set(
                    all_items) - items_interacted_set - set(items_assumed_to_be_interacted)
                true_positive = 0
                false_positive = 0
                true_negative = 0
                false_negative = 0
                for interacted_item in items_interacted_set:
                    if interacted_item in items_recommended_set:
                        true_positive += 1  # If the item is interacted and it is recommended
                    else:
                        false_negative += 1  # If the item is interacted and it is NOT recommended
                for recommend_item in items_recommended_set:
                    if recommend_item not in items_interacted_set:
                        false_positive += 1  # If the item is recommended and it is NOT interacted
                for non_interacted_item in items_not_interacted_set:
                    if non_interacted_item in items_recommended_set:
                        false_positive += 1  # If the item is NOT interacted and it is recommended
                    else:
                        true_negative += 1
                # precision is the proportion of recommendations that are good
                # recommendations
                # precision = float(len(hitset)) / no_of_items_to_recommend
                # recall is the proportion of good recommendations that appear
                # in top recommendations
                # recall = float(len(hitset)) / no_of_items_interacted
                #print("earlier precision : {} recall : {}".format(precision, recall))
                # print("tp : {} fp : {} tn : {} fn : {}".format(true_positive,
                #                                                false_positive,
                #                                                true_negative,
                #                                                false_negative))
                true_positive_false_positive = true_positive + false_positive
                if true_positive_false_positive != 0:
                    precision = float(true_positive) / true_positive_false_positive
                else:
                    precision = 0.0
                true_positive_false_negative = true_positive + false_negative
                if true_positive_false_negative != 0:
                    recall = float(true_positive) / true_positive_false_negative
                else:
                    recall = 0.0
                #print("now precision : {} recall : {}".format(precision, recall))
                if (recall + precision) != 0:
                    f1_score = float(2 * precision * recall) / (recall + precision)
                else:
                    f1_score = 0.0
                mcc_numerator = (float(true_positive * true_negative) - \
                                 float(false_positive * false_negative))

                true_negative_false_positive = true_negative + false_positive
                true_negative_false_negative = true_negative + false_negative
                mcc_denominator = math.sqrt((true_positive_false_positive) * \
                                            (true_positive_false_negative) * \
                                            (true_negative_false_positive) * \
                                            (true_negative_false_negative))
                if mcc_denominator == 0 :
                    mcc_denominator = 1#https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
                mcc_score = mcc_numerator / mcc_denominator
                #print("f1_score : {} mcc : {}".format(f1_score, mcc_score))
                
                sum_precision += precision
                sum_recall += recall
                sum_f1_score += f1_score
                sum_mcc_score += mcc_score

            avg_precision = sum_precision / float(num_users_sample)
            avg_recall = sum_recall / float(num_users_sample)
            avg_f1_score = sum_f1_score / float(num_users_sample)
            avg_mcc_score = sum_mcc_score / float(num_users_sample)

            results['no_of_items_to_recommend'][no_of_items_to_recommend][
                'avg_precision'] = round(avg_precision, 4)
            results['no_of_items_to_recommend'][no_of_items_to_recommend][
                'avg_recall'] = round(avg_recall, 4)
            results['no_of_items_to_recommend'][no_of_items_to_recommend][
                'avg_f1_score'] = round(avg_f1_score, 4)
            results['no_of_items_to_recommend'][no_of_items_to_recommend][
                'avg_mcc_score'] = round(avg_mcc_score, 4)

        return results

    def compute_precision_recall1(self, eval_items_grps):
        """calculate the precision and recall measures"""
        # For each distinct no_of_items:
        #    1. For each distinct user, calculate precision and recall.
        #    2. Calculate average precision and recall.
        results = {}

        no_of_items_to_recommend_list = eval_items_grps.keys()
        results['no_of_items_to_recommend'] = dict()
        for no_of_items_to_recommend in no_of_items_to_recommend_list:
            results['no_of_items_to_recommend'][
                no_of_items_to_recommend] = dict()

            eval_items = eval_items_grps[no_of_items_to_recommend]
            users_test_sample = eval_items.keys()
            num_users_sample = len(users_test_sample)

            sum_precision = 0
            sum_recall = 0
            sum_f1_score = 0
            avg_precision = 0
            avg_recall = 0

            for user_id in users_test_sample:
                items_interacted = eval_items[user_id]['items_interacted']
                items_recommended = eval_items[user_id][
                    'items_recommended'][0:no_of_items_to_recommend]

                hitset = set(items_interacted).intersection(
                    set(items_recommended))
                testset = items_interacted
                no_of_items_interacted = len(testset)
                # if len(hitset) > 0:
                #     print("User ID : ", user_id)
                #     print("Items Interacted : ", set(items_interacted))
                #     print("Items Recommended: ", set(items_recommended))
                #     print("Hitset : ", hitset)
                #     input()

                # precision is the proportion of recommendations that are good
                # recommendations
                precision = float(len(hitset)) / no_of_items_to_recommend
                # recall is the proportion of good recommendations that appear
                # in top recommendations
                recall = float(len(hitset)) / no_of_items_interacted
                if (recall + precision) != 0:
                    f1_score = float(2 * precision * recall) / \
                        (recall + precision)
                else:
                    f1_score = 0.0
                sum_precision += precision
                sum_recall += recall
                sum_f1_score += f1_score

            avg_precision = sum_precision / float(num_users_sample)
            avg_recall = sum_recall / float(num_users_sample)
            avg_f1_score = sum_f1_score / float(num_users_sample)

            results['no_of_items_to_recommend'][no_of_items_to_recommend][
                'avg_precision'] = round(avg_precision, 4)
            results['no_of_items_to_recommend'][no_of_items_to_recommend][
                'avg_recall'] = round(avg_recall, 4)
            results['no_of_items_to_recommend'][no_of_items_to_recommend][
                'avg_f1_score'] = round(avg_f1_score, 4)

        return results
