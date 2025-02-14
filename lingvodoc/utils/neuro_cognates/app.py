import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Layer
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
import itertools
import json
import os
import gzip
import pickle
import multiprocess
from time import time as now
from lingvodoc.queue.celery import celery
from lingvodoc.cache.caching import TaskStatus, initialize_cache


class AbsDiffLayer(Layer):
    def call(self, inputs):
        x1, x2 = inputs
        return tf.math.abs(x1 - x2)

    def get_config(self):
        config = super(AbsDiffLayer, self).get_config()
        return config


@celery.task
def predict_cognates(
        word_pairs,
        task_key,
        cache_kwargs,
        compare_lists,
        input_index,
        tokenizer,
        model,
        max_len,
        stamp,
        perspective_name_list,
        storage,
        four_tensors=False,
        truth_threshold=0.97):

    def split_items(items):
        return (
            list(map(lambda x: x[0], items)),
            list(map(lambda x: x[1], items)),
            list(map(lambda x: x[2], items)))

    # Разделяем входные пары на слова и переводы
    input_words, input_translations, input_lex_ids = split_items(word_pairs)

    # Токенизация и паддинг входных данных
    seq_input_words = [tokenizer.texts_to_sequences([word]) for word in input_words]
    X_input_words = [pad_sequences(seq, maxlen=max_len, padding='post') for seq in seq_input_words]
    X_input_translations = []

    if four_tensors:
        seq_input_translations = [tokenizer.texts_to_sequences([trans]) for trans in input_translations]
        X_input_translations = [pad_sequences(seq, maxlen=max_len, padding='post') for seq in seq_input_translations]

    X_compare_words = []
    X_compare_translations = []

    # Проход по каждому списку для сравнения
    for compare_list in compare_lists:

        compare_words, compare_translations, compare_lex_ids = split_items(compare_list)

        # Токенизация и паддинг данных для сравнения
        seq_compare_words = [tokenizer.texts_to_sequences([word]) for word in compare_words]
        X_compare_words.append([pad_sequences(seq, maxlen=max_len, padding='post') for seq in seq_compare_words])

        if four_tensors:
            seq_compare_translations = [tokenizer.texts_to_sequences([trans])
                                        for trans in compare_translations]
            X_compare_translations.append([pad_sequences(seq, maxlen=max_len, padding='post')
                                          for seq in seq_compare_translations])
        else:
            X_compare_translations.append([])

    stamp_file = f"/tmp/lingvodoc_stamps/{stamp}"


    # Calculate prediction
    def get_prediction(input_word, input_trans, input_id, X_word, X_trans):

        similarities = []
        result = []

        # Проход по каждому списку для сравнения
        for i, compare_list in enumerate(compare_lists):

            if not compare_list:
                continue

            compare_words, compare_translations, compare_lex_ids = split_items(compare_list)

            count = 0
            for compare_word, compare_trans, compare_id, X_comp_word, X_comp_trans in itertools.zip_longest(
                compare_words, compare_translations, compare_lex_ids, X_compare_words[i], X_compare_translations[i]):

                # Checking stamp-to-stop every hundred comparings
                count += 1
                if count % 100 == 0 and os.path.isfile(stamp_file):
                    print("Killed process !!!")
                    #return result

                # Передаем 2 или 4 тензора в модель
                pred = (model.predict([X_word, X_trans, X_comp_word, X_comp_trans])[0][0]
                        if four_tensors else
                        model.predict([X_word, X_comp_word])[0][0])

                if pred > truth_threshold:  # Фильтр по вероятности
                    similarities.append((i, [compare_word, compare_trans], compare_id, f"{pred:.4f}"))

            if similarities:
                result.append((
                    input_index,
                    f"{input_word} '{input_trans}'",
                    input_id,
                    None,
                    similarities,
                    []))

        return result


    start_time = now()
    results = []
    current_stage = 0
    flushed = 0
    input_len = len(word_pairs)
    compare_len = sum(map(len, compare_lists))
    initialize_cache(cache_kwargs)
    task = TaskStatus.get_from_cache(task_key)


    def add_result(res):
        nonlocal current_stage, flushed
        current_stage += 1
        passed = now() - start_time
        left = passed / current_stage * input_len - passed

        days = int(left / 86400)
        hours = int((left - days * 86400) / 3600)
        minutes = int((left - days * 86400 - hours * 3600) / 60)

        result_link = ""
        progress = int(current_stage / input_len * 100)
        status = f">> {days}d:{hours}h:{minutes}m left <<"

        results.extend(res)

        if passed - flushed > 300 or current_stage == input_len:
            flushed = passed

            result_dict = (
                dict(
                    suggestion_list = results,
                    perspective_name_list = perspective_name_list,
                    transcription_count = compare_len * current_stage))

            storage_dir = os.path.join(storage['path'], 'neuro_cognates')
            pickle_path = os.path.join(storage_dir, str(stamp))
            os.makedirs(storage_dir, exist_ok=True)

            with gzip.open(pickle_path, 'wb') as result_data_file:
                pickle.dump(result_dict, result_data_file)

            result_link = ''.join([
                storage['prefix'],
                storage['static_route'],
                'suggestions/',
                str(stamp)
            ])

            if current_stage == input_len:
                progress = 100
                status = "Finished"

        task.set(current_stage, progress, status, result_link)



    with multiprocess.Pool(multiprocess.cpu_count() // 2) as p:
        for args in itertools.zip_longest(
            input_words,
            input_translations,
            input_lex_ids,
            X_input_words,
            X_input_translations
        ):

            p.apply_async(get_prediction, args, callback=add_result)

        p.close()
        p.join()

    # Removing stamp-to-stop if exists
    try:
        os.remove(stamp_file)
    except OSError:
        pass

    return results


class NeuroCognates:
    def __init__(self,
                 compare_lists,
                 input_index,
                 four_tensors,
                 truth_threshold,
                 stamp,
                 perspective_name_list,
                 storage):

        self.compare_lists = compare_lists
        self.input_index = input_index
        self.four_tensors = four_tensors
        self.truth_threshold = truth_threshold
        self.stamp = stamp
        self.perspective_name_list = perspective_name_list
        self.storage = storage

        project_dir = os.path.abspath(os.getcwd())
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        os.chdir(script_dir)

        if four_tensors:
            try:
                with open('config_dict.json', 'r', encoding='utf-8') as config_file:
                    config_dict = json.load(config_file)
            except FileNotFoundError:
                print("Файл config_dict.json не найден. Убедитесь, что файл находится в корневой директории проекта.")
                config_dict = {}
            except json.JSONDecodeError:
                print("Ошибка чтения файла config_dict.json. Проверьте корректность JSON.")
                config_dict = {}

            self.max_len_dict = config_dict.get('sequence_length')
            if self.max_len_dict is None:
                print("sequence_length не найден в config_dict.json. Установлено значение по умолчанию 100.")
                self.max_len_dict = 100

            self.model_dict = tf.keras.models.load_model(
                'my_model_dict.keras',
                custom_objects={'AbsDiffLayer': AbsDiffLayer}
            )

            with open('tokenizer_dict.json', 'r', encoding='utf-8') as f:
                tokenizer_dict_data = json.load(f)
            self.tokenizer_dict = tokenizer_from_json(tokenizer_dict_data)

        else:

            # Загрузка конфигурации из файла config.json
            try:
                with open('config99.json', 'r', encoding='utf-8') as config_file:
                    config = json.load(config_file)
            except FileNotFoundError:
                print("Файл config.json не найден. Убедитесь, что файл находится в корневой директории проекта.")
                config = {}
            except json.JSONDecodeError:
                print("Ошибка чтения файла config.json. Проверьте корректность JSON.")
                config = {}

            self.max_len = config.get('sequence_length')
            if self.max_len is None:
                print("sequence_length не найден в config.json. Установлено значение по умолчанию 100.")
                self.max_len = 100

            self.model = tf.keras.models.load_model(
                'my_model.keras',
                custom_objects={'AbsDiffLayer': AbsDiffLayer}
            )

            with open('tokenizer.json', 'r', encoding='utf-8') as f:
                tokenizer_data = json.load(f)
            self.tokenizer = tokenizer_from_json(tokenizer_data)

        # Change dir back
        os.chdir(project_dir)

    def index(self, word_pairs, task_key, cache_kwargs):

        if self.four_tensors:
            # Вызов функции для сравнения (модель с 4 тензорами)
            return predict_cognates.delay(
                word_pairs,
                task_key,
                cache_kwargs,
                self.compare_lists,
                self.input_index,
                self.tokenizer_dict,
                self.model_dict,
                self.max_len_dict,
                self.stamp,
                self.perspective_name_list,
                self.storage,
                self.four_tensors,
                self.truth_threshold)
        else:
            # Вызов функции для сравнения (модель с 2 тензорами)
            return predict_cognates.delay(
                word_pairs,
                task_key,
                cache_kwargs,
                self.compare_lists,
                self.input_index,
                self.tokenizer,
                self.model,
                self.max_len,
                self.stamp,
                self.perspective_name_list,
                self.storage,
                self.four_tensors,
                self.truth_threshold)
