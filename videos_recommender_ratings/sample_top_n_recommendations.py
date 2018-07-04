from collections import defaultdict

from surprise import SVD
from surprise import Dataset


def get_top_n(predictions, n=10):
    '''Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

def get_testset_stats(testset):
    #print("Testing Data")
    test_users = []
    test_items = []
    test_ratings = []
    for (user_id, item_id, rating) in testset:
        #print((user_id, item_id, rating))
        test_users.append(user_id)
        test_items.append(item_id)
        test_ratings.append(rating)
    test_users_set = set(test_users)
    test_items_set = set(test_items)
    #print("No of users : {}, users_set : {}".format(len(test_users), len(test_users_set)))
    #print("No of items : {}, items_set : {}".format(len(test_items), len(test_items_set)))
    #print("No of ratings : {}".format(len(test_ratings)))
    return len(test_users_set),  len(test_items_set), len(test_ratings)

# First train an SVD algorithm on the movielens dataset.
data = Dataset.load_builtin('ml-100k')
trainset = data.build_full_trainset()
testset = trainset.build_anti_testset()

print("trainset : ", trainset.n_users, trainset.n_items, trainset.n_ratings)
testset_n_users, testset_n_items, testset_n_ratings = get_testset_stats(testset)
print("testset  : ", testset_n_users, testset_n_items, testset_n_ratings)

algo = SVD()
algo.fit(trainset)

# Than predict ratings for all pairs (u, i) that are NOT in the training set.

predictions = algo.test(testset)
top_n = get_top_n(predictions, n=10)

# Print the recommended items for each user
for uid, user_ratings in top_n.items():
    print(uid, [iid for (iid, _) in user_ratings])
