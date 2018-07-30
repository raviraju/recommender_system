"""Module to perform translation of korean metadata"""
import os
import json
import argparse
import pandas as pd

TRANSLATIONS_FILE = 'translations.json'

def load_json_data(input_file):
    """load json data"""
    data = None
    if os.path.isfile(input_file):
        with open(input_file, 'r', encoding='utf-8') as json_file:
            data = json.load(fp=json_file)
    return data

def dump_json_data(data, output_file):
    """dump json data"""
    print("Dump json into ", output_file)
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(data, fp=json_file, indent=4)

def convert_korean_text(book_meta_data, korean_field):
    """method to use google translate api to translate the field"""
    print("convert_korean_text : ", korean_field)
    if os.path.isfile(TRANSLATIONS_FILE):
        translations_mapping = load_json_data(TRANSLATIONS_FILE)
    else:
        translations_mapping = dict()

    values = book_meta_data[korean_field]
    translated_values = []
    for val in values:
        if isinstance(val, str):
            tokens = set()
            if ':' in val:
                components = (val.split(':'))
                for component in components:
                    tokens.add(component)
            elif ',' in val:
                components = (val.split(':'))
                for component in components:
                    tokens.add(component)
            elif '.' in val:
                components = (val.split(':'))
                for component in components:
                    tokens.add(component)
            elif '/' in val:
                components = (val.split(':'))
                for component in components:
                    tokens.add(component)
            else:
                #print(val)
                tokens.add('')
                tokens.add(val)
                #print(tokens)
                #input()
            translated_tokens = set()
            for token in tokens:                
                translated_token = translations_mapping.get(token, "")
                #print("token : {}, translated_token : {}".format(token, translated_token))
                #input()
                if(len(translated_token) > 1):#to avoid blank
                    translated_tokens.add(translated_token)
            translated_val = '|'.join(translated_tokens)
            #print("translated_val : ", translated_val)
        else:
            translated_val = ''
        # print(val)
        # print(translated_val)
        # input()
        translated_values.append(translated_val)

    translated_korean_field = 'T_' + korean_field
    book_meta_data[translated_korean_field] = pd.Series(translated_values)
    return book_meta_data

def preprocess_values(unique_values):
    """preprocess korean text to get tokens"""
    values = set()
    for val in unique_values:
        if ':' in val:
            components = (val.split(':'))
            for component in components:
                values.add(component)
        elif ',' in val:
            components = (val.split(':'))
            for component in components:
                values.add(component)
        elif '.' in val:
            components = (val.split(':'))
            for component in components:
                values.add(component)
        elif '/' in val:
            components = (val.split(':'))
            for component in components:
                values.add(component)
        else:
            values = values.union(val)
    return list(values)

def store_korean_text(book_meta_data, korean_dir, korean_field, preprocess=False):
    """store text from korean fields into respective json files"""
    print("store_korean_text : ", korean_field)
    korean_text_file = os.path.join(korean_dir, korean_field + '.json')
    data = load_json_data(korean_text_file)
    if not data:
        input_set = set()
    else:
        input_set = set(data)

    values = book_meta_data[korean_field]
    unique_values = values.unique()
    unique_values = unique_values[~pd.isnull(unique_values)]
    if preprocess:
        unique_values = preprocess_values(unique_values)
    values_to_translate = set(list(unique_values))

    input_set = input_set.union(values_to_translate)
    print("No of items : ", len(input_set))
    dump_json_data(list(input_set), korean_text_file)

def main():
    """interface to get translations for korean fields"""
    parser = argparse.ArgumentParser(description="Translations for korean fields")
    parser.add_argument("--store",
                        help="Store Korean Text",
                        action="store_true")
    parser.add_argument("--convert",
                        help="Convert Korean Text",
                        action="store_true")
    args = parser.parse_args()

    data_csv_dir = '../data/'
    data_csv = 'BOOKMETA.csv'
    data_csv_file = os.path.join(data_csv_dir, data_csv)
    print("loading {}...".format(data_csv_file))
    book_meta_data = pd.read_csv(data_csv_file)
    config = {'BOOK_NAME': {'preprocess':False},
              'KEYWORD': {'preprocess':True},
              'AUTHOR': {'preprocess':False}
              #'SUMMARY': {'preprocess':False}
             }
    if args.store:
        korean_dir = "korean_dir"
        if not os.path.isdir(korean_dir):
            os.makedirs(korean_dir)
        for korean_field in config:
            preprocess = config[korean_field]['preprocess']
            store_korean_text(book_meta_data, korean_dir, korean_field, preprocess)
    elif args.convert:
        for korean_field in config:
            book_meta_data = convert_korean_text(book_meta_data, korean_field)
            translated_korean_field = 'T_' + korean_field
            print(book_meta_data[[korean_field, translated_korean_field]].head())
        t_data_csv_file = os.path.join('../data', 'T_' + data_csv)
        print("storing {}...".format(t_data_csv_file))
        book_meta_data.to_csv(t_data_csv_file, index=False)
    else:
        print("Invalid option")

if __name__ == '__main__':
    main()
