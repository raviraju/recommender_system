NOTE : Download dataset from the following urls and place in `data` directory

10000.txt : https://static.turi.com/datasets/millionsong/10000.txt
song_data.csv : https://static.turi.com/datasets/millionsong/song_data.csv

Popularity Based Recommendation
```
$ python popularity_based.py
No. of users in the training set: 365
No. of items in the training set: 4483
Training Completed in :                               Time Taken : 0.1024    sec
Recommendations generated in :                        Time Taken : 0.0376    sec
Popularity Based Recommendations are found in results/
```

Item Based Colloborative Filtering Recommendation
```
$ python item_based_cf.py
No. of users in the training set: 365
No. of items in the training set: 4483
No. of items for the user_id 97e48f0f188e04dcdb0f8e20a29dacb881d80c9e : 104
Non zero values in Co-Occurence_matrix : 17904
Density : 0.038401482523722094
Computing CoOccurence Matrix Completed in :           Time Taken : 6.0697    sec
Recommendations generated in :                        Time Taken : 0.0553    sec
Item Based Recommendations are found in results/
```