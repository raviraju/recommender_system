### Store all korean fields to be converted in json files in korean_dir/
$ `python book_meta.py --store`

### Obtain google translations for json files in korean_dir, update translations in translation_dir/translations.json
$ `python translation_api.py`

### Convert korean fields using google translations in translation_dir/translations.json to produce output/T_BOOKMETA.csv
$ `python book_meta.py --convert`

### Merge T_BOOKMETA.csv and BOOKINFORMATION.csv to produce output/BOOKINFORMATION_META.csv
$ `python merge_book_info_meta.py`

