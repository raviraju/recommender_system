# [Recommender System](documentation/about_recommenders.md)

This repository serves as a framework to build Recommendation Engine in Python.
It supports two types of recommender systems based on availability of user item ratings
* Binary  Recommender : Leverages users past behaviour in form of binary data (0/1), representing user interaction with item
* Ratings Recommender : Leverages users provided ratings for items they have interacted
    * Utilizes algorithms from [Surprise](http://surpriselib.com/) - a Python scikit library for building and analyzing rating based recommender systems,
    * Integrates use of tracking component of [MLFlow](https://mlflow.org/) - A Platform for ML LifeCycle, in order log and track various hyperparameter tuning and model training experiments.

## Binary Recommender

- ### [Books Recommender](https://github.com/raviraju/recommender_system/blob/master/binary_recommender/books_recommender)

    Data : Woongjin Book Club Dataset provided by Kidaptive.

    [Project Overview](https://github.com/raviraju/recommender_system/blob/master/binary_recommender/books_recommender/Recommender%20Woongjin%20.pdf)
    provides a description of the dataset, data preprocessing steps, different types of colloborative and content based recommender systems built,
    along with evaluation results for comparision.

    [Project Readme](https://github.com/raviraju/recommender_system/blob/master/binary_recommender/books_recommender/readme.md) describes
    the steps used to leverage various python modules to build project pipeline.

- ### [Articles Recommender](https://github.com/raviraju/recommender_system/blob/master/binary_recommender/articles_recommender)

    Data : Articles shared among companies employees provided by [CI&T's](https://us.ciandt.com/) [Deskdrop](https://deskdrop.co/) platform

    [Project Readme](https://github.com/raviraju/recommender_system/blob/master/binary_recommender/articles_recommender/readme.md) describes
    the steps used to leverage various python modules to build project pipeline.

## Ratings Recommender

- ### [Media Recommender](https://github.com/raviraju/recommender_system/tree/master/ratings_recommender)

    Data : Compass Dataset provided by Kidaptive. 

    [Project Overview](https://github.com/raviraju/recommender_system/blob/master/ratings_recommender/Compass_Recommender.pdf)
provides a description of the dataset, data preprocessing steps, different recommender algorithms utilized,
along with evaluation results comparing various recommenders built with existing recommender at Kidaptive.

    [Project Readme](https://github.com/raviraju/recommender_system/blob/master/ratings_recommender/readme.md) describes
the steps used to leverage various python modules to build project pipeline.
