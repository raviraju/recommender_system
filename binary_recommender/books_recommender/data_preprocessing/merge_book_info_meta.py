import pandas as pd

book_meta = pd.read_csv('../data/T_BOOKMETA.csv')
book_info = pd.read_csv('../data/BOOKINFORMATION.csv')

book_meta_info = pd.merge(book_meta,
                          book_info,
                          how='inner',
                          on='BOOK_META_CODE',
                          suffixes=('_BM', '_BI'))

book_meta_info.to_csv('../data/BOOKINFORMATION_META.csv', index=False)
