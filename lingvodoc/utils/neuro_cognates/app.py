import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Layer
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
import itertools
import json
import os
import multiprocess
from pdb import set_trace as A


class AbsDiffLayer(Layer):
    def call(self, inputs):
        x1, x2 = inputs
        return tf.math.abs(x1 - x2)

    def get_config(self):
        config = super(AbsDiffLayer, self).get_config()
        return config


class NeuroCognates:
    def __init__(self, four_tensors, truth_threshold):

        self.four_tensors = four_tensors
        self.truth_threshold = truth_threshold

        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)

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

    @staticmethod
    def split_items(items):
        return (
            list(map(lambda x: x[0], items)),
            list(map(lambda x: x[1], items)),
            list(map(lambda x: x[2], items)))

    @staticmethod
    def predict_cognates(
            word_pairs,
            compare_lists,
            input_index,
            tokenizer,
            model,
            max_len,
            four_tensors=False,
            truth_threshold=0.97):

        # Разделяем входные пары на слова и переводы
        input_words, input_translations, input_lex_ids = NeuroCognates.split_items(word_pairs)

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

            compare_words, compare_translations, compare_lex_ids = NeuroCognates.split_items(compare_list)

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

        # Calculate prediction
        def get_prediction(input_word, input_trans, input_id, X_word, X_trans):

            similarities = []
            result = []

            # Проход по каждому списку для сравнения
            for i, compare_list in enumerate(compare_lists):

                if not compare_list:
                    continue

                compare_words, compare_translations, compare_lex_ids = NeuroCognates.split_items(compare_list)

                for compare_word, compare_trans, compare_id, X_comp_word, X_comp_trans in itertools.zip_longest(
                    compare_words, compare_translations, compare_lex_ids, X_compare_words[i], X_compare_translations[i]):

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

        with multiprocess.Pool(multiprocess.cpu_count() * 2) as p:

            results = p.starmap(get_prediction, itertools.zip_longest(
                input_words, input_translations, input_lex_ids, X_input_words, X_input_translations))

            plain_results = []
            for result in results:
                plain_results.extend(result)

            p.close()
            p.join()

        return plain_results

    def index(self, word_pairs, compare_lists, input_index):
        if self.four_tensors:
            # Вызов функции для сравнения (модель с 4 тензорами)
            return NeuroCognates.predict_cognates(
                word_pairs,
                compare_lists,
                input_index,
                self.tokenizer_dict,
                self.model_dict,
                self.max_len_dict,
                self.four_tensors,
                self.truth_threshold)
        else:
            # Вызов функции для сравнения (модель с 2 тензорами)
            return NeuroCognates.predict_cognates(
                word_pairs,
                compare_lists,
                input_index,
                self.tokenizer,
                self.model,
                self.max_len,
                self.four_tensors,
                self.truth_threshold)
