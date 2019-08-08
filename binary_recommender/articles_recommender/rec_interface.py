"""Module for Articles Recommender"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                #binary_rec     #articles_recommender

import logging
logging.basicConfig(level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)

from recommender import rec_interface as generic_rec_interface

class ArticlesRecommender(generic_rec_interface.Recommender):
    """encapsulating common functionality for articles recommender use case"""
    def __init__(self, results_dir, model_dir,
                 train_data, test_data,
                 user_id_col, item_id_col, **kwargs):
        """constructor"""
        super().__init__(results_dir, model_dir,
                         train_data, test_data,
                         user_id_col, item_id_col, **kwargs)

    def derive_stats(self):
        """derive use case specific stats"""
        super().derive_stats()

        LOGGER.debug("Test Data        :: Getting User Groups")