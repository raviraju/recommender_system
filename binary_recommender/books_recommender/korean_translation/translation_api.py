"""Module to perform google translation on a input list"""
import json
import os
import time
import fnmatch

from google.cloud import translate
#Get service account key from google translation cloud account
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'translation_dir/Translate-df698c42870a.json'

TRANSLATIONS_FILE = 'translations.json'
FAILED_TRANSLATIONS_FILE = 'korean_dir/translations_failed.json'
SLEEP_TIME = 10
BATCH_SIZE = 50

def load_json_data(input_file):
    """load json data"""
    with open(input_file, 'r', encoding='utf-8') as json_file:
        data = json.load(fp=json_file)
    return data

def dump_json_data(data, output_file):
    """dump json data"""
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(data, fp=json_file, indent=4)

def get_chunks(vals, chunk_size):
    """Yield successive chunk_size from vals."""
    for i in range(0, len(vals), chunk_size):
        yield vals[i:i + chunk_size]

def get_google_translation(values_to_translate):
    """get translation of given values"""
    # Wait for sleep_time seconds
    print("\nWaiting for {} secs ...".format(SLEEP_TIME), end='')
    time.sleep(SLEEP_TIME)
    try:
        # Instantiates a client
        client = translate.Client(target_language='en')
        print("\nFetching Translations...")
        mapping = dict()
        translations = client.translate(values_to_translate)
        for translation in translations:
            input_txt = translation['input']
            translated_txt = translation['translatedText']
            #detected_src_lang = translation['detectedSourceLanguage']
            mapping[input_txt] = translated_txt
            print(input_txt, ' -> ', translated_txt)
        return mapping
    except Exception as err:
        print("Failed in fetching translations..retry after a while")
        print(err)
        return None

def update_translations_mapping(translations_mapping, mapping):
    """update_translations_mapping with given mapping"""
    for key, val in mapping.items():
        translations_mapping[key] = val
    #saving updated translations
    dump_json_data(translations_mapping, TRANSLATIONS_FILE)

def get_no_of_chars(translations_sample):
    no_of_chars = 0
    for sample in translations_sample:
        #print(sample, len(sample))
        no_of_chars += len(sample)
    return no_of_chars

def main():
    if os.path.isfile(TRANSLATIONS_FILE):
        translations_mapping = load_json_data(TRANSLATIONS_FILE)
    else:
        translations_mapping = dict()

    korean_dir = "korean_dir"
    if not os.path.isdir(korean_dir):
        print("Ensure to have korean text json files in :", korean_dir)
        return

    pattern = '*.json'
    for root, dirs, files in os.walk(korean_dir):
        for filename in fnmatch.filter(files, pattern):
            korean_file = (os.path.join(root, filename))

            if os.path.isfile(FAILED_TRANSLATIONS_FILE):
                failed_translations = load_json_data(FAILED_TRANSLATIONS_FILE)
            else:
                failed_translations = []

            print("Translating : ", korean_file)
            #load korean text to translate
            input_list = load_json_data(korean_file)
            unknown_translations = []
            for val in input_list:
                if val not in translations_mapping:
                    unknown_translations.append(val)
            no_of_unknown_translations = len(unknown_translations)
            print("no_of_unknown_translations : ", no_of_unknown_translations)

            #get translations in a batch
            translations_samples = list(get_chunks(unknown_translations, BATCH_SIZE))
            for translations_sample in translations_samples:
                #print(len(translations_sample))
                no_of_items = len(translations_sample)
                no_of_chars = get_no_of_chars(translations_sample)
                print("No of items : {}, No of Chars : {}".format(no_of_items, no_of_chars))
                mapping = get_google_translation(translations_sample)
                if mapping:
                    update_translations_mapping(translations_mapping, mapping)
                else:
                    failed_translations.extend(translations_sample)
                    break

            #saving failed translations
            dump_json_data(failed_translations, FAILED_TRANSLATIONS_FILE)

if __name__ == '__main__':
    main()

#export GOOGLE_APPLICATION_CREDENTIALS='Translate-df698c42870a.json'
#echo $GOOGLE_APPLICATION_CREDENTIALS
#/home/ravi/Desktop/kidaptive/git_files/Translate-df698c42870a.json