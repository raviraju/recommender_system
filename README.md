# Recommender System

Recommendation Engines are a subclass of information filtering system that seek to predict the item preference for an user.
The objective of a [Recommender System](https://en.wikipedia.org/wiki/Recommender_system) is to recommend relevant items for users, based on their preference. Preference and relevance are subjective, and they are generally inferred by items users have consumed previously.

The main families of methods for RecSys are:
- [Collaborative Filtering](https://en.wikipedia.org/wiki/Collaborative_filtering): This method makes automatic predictions (filtering) about the interests of a user by collecting preferences or taste information from many users (collaborating). The underlying assumption of the collaborative filtering approach is that if a person A has the same opinion as a person B on a set of items, A is more likely to have B's opinion for a given item than that of a randomly chosen person.
    - [Collaborative Filtering User-Based Overview](documentation/cbf-user-based.png)
    - [Collaborative Filtering Item-Based Overview](documentation/cbf-item-based.png)
        - [Item-item collaborative filtering](https://en.wikipedia.org/wiki/Item-item_collaborative_filtering), or item-based, or item-to-item, is a form of collaborative filtering for recommender systems based on the similarity between items calculated using people's ratings of those items.
        - Earlier collaborative filtering systems based on rating similarity between users (known as user-user collaborative filtering) had several problems:
            - systems performed poorly when they had many items but comparatively few ratings
            - computing similarities between all pairs of users was expensive
            - user profiles changed quickly and the entire system model had to be recomputed
        - Item-item models resolve these problems in systems that have more users than items.
        - It uses the most similar items to a user's already-rated items to generate a list of recommendations. 
        - the similarities are based on [correlations between the purchases of items by users](https://patents.google.com/patent/US6266649) (e.g., items A and B are similar because a relatively large portion of the users that purchased item A also bought item B).
        - This form of recommendation is analogous to "people who rate item X highly, like you, also tend to rate item Y highly, and you haven't rated item Y yet, so you should try it".
        - [Among CF, Item-based CF (IBCF) is a well-known technique that provides accurate recommendations and has been used by Amazon as well.]()
    - [Collaborative Filtering Pros & Cons](documentation/cbf-pros_cons.png)
    - [if No. of items is greater than No. of users go with user-based collaborative filtering as it will reduce the computation power and If No. of users is greater than No. of items go with item-based collaborative filtering. For Example, Amazon has lakhs of items to sell but has billions of customers.](https://www.analyticsvidhya.com/blog/2021/07/recommendation-system-understanding-the-basic-concepts/). [Hence Amazon uses item-based collaborative filtering because of less no. of products as compared to its customers.](https://www.cs.umd.edu/~samir/498/Amazon-Recommendations.pdf)

- [Content-Based Filtering](http://recommender-systems.org/content-based-filtering/): This method uses only information about the description and attributes of the items users has previously consumed to model user's preferences. In other words, these algorithms try to recommend items that are similar to those that a user liked in the past (or is examining in the present). In particular, various candidate items are compared with items previously rated by the user and the best-matching items are recommended. Good at finding items that are similar to other items preferred by user, but not so hot at finding something new.
    - [Content Based Overview](documentation/content-based.png)
    - [Content Based Pros & Cons](documentation/content-based_pros_cons.png)
    - [Content-based methods seem to suffer far less from the cold start problem](https://analyticsindiamag.com/collaborative-filtering-vs-content-based-filtering-for-recommender-systems/) than collaborative approaches because new users or items can be described by their characteristics i.e the content and so relevant suggestions can be done for these new entities. Only new users or items with previously unseen features will logically suffer from this drawback, but once the system is trained enough, this has little to no chance to happen. Basically, it hypothesizes that if a user was interested in an item in the past, they will once again be interested in the same thing in the future. 

- [Collaborative Filtering Vs Content-Based Filtering](https://analyticsindiamag.com/collaborative-filtering-vs-content-based-filtering-for-recommender-systems/) [Paper](https://arxiv.org/pdf/1912.08932.pdf)

- Hybrid methods: Recent research has demonstrated that a hybrid approach, combining collaborative filtering and content-based filtering could be more effective than pure approaches in some cases. These methods can also be used to overcome some of the common problems in recommender systems such as cold start and the sparsity problem.

To know more about state-of-the-art methods published in Recommender Systems on [ACM RecSys conference](https://recsys.acm.org/).


This repository serves as a framework to build Recommendation Engine in Python. 
It supports two types of recommender systems based on availability of user item ratings
* Binary  Recommender : Users past behaviour in form of binary data (0/1), representing user interaction with item
* Ratings Recommender : Users provided ratings for items they have interacted

## Binary Recommender

* [Books Recommender](https://github.com/raviraju/recommender_system/blob/master/binary_recommender/books_recommender)  
    
    Data : Woongjin Book Club Dataset provided by Kidaptive.

    [Project Overview](https://github.com/raviraju/recommender_system/blob/master/binary_recommender/books_recommender/Recommender%20Woongjin%20.pdf)
    provides a description of the dataset, data preprocessing steps, different types of colloborative and content based recommender systems built,
    along with evaluation results for comparision.

    [Project Readme](https://github.com/raviraju/recommender_system/blob/master/binary_recommender/books_recommender/readme.md) describes
    the steps used to leverage various python modules to build project pipeline.

* [Articles Recommender](https://github.com/raviraju/recommender_system/blob/master/binary_recommender/articles_recommender)

    Data : Articles shared among companies employees provided by [CI&T's](https://us.ciandt.com/) [Deskdrop](https://deskdrop.co/) platform

    [Project Readme](https://github.com/raviraju/recommender_system/blob/master/binary_recommender/articles_recommender/readme.md) describes
    the steps used to leverage various python modules to build project pipeline.

## Ratings Recommender

* About
    * Utilizes algorithms from [Surprise](http://surpriselib.com/) - a Python scikit library for building and analyzing rating based recommender systems,
    * Integrates use of Tracking component of [MLFlow](https://mlflow.org/) - A Platform for ML LifeCycle, in order log and track
various hyperparamter tuning and model training experiments.

* [Media Recommender](https://github.com/raviraju/recommender_system/tree/master/ratings_recommender)

    Data : Compass Dataset provided by Kidaptive. 

    [Project Overview](https://github.com/raviraju/recommender_system/blob/master/ratings_recommender/Compass_Recommender.pdf)
provides a description of the dataset, data preprocessing steps, different recommender algorithms utilized,
along with evaluation results comparing various recommenders built with existing recommender at Kidaptive.

    [Project Readme](https://github.com/raviraju/recommender_system/blob/master/ratings_recommender/readme.md) describes
the steps used to leverage various python modules to build project pipeline.
