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
import numpy as np
from time import time as now
from lingvodoc.queue.celery import celery
from lingvodoc.cache.caching import TaskStatus, initialize_cache


class AbsDiffLayer(Layer):
    @staticmethod
    def call(inputs):
        x1, x2 = inputs
        return tf.math.abs(x1 - x2)

    def get_config(self):
        config = super(AbsDiffLayer, self).get_config()
        return config


class NeuroCognates:
    def __init__(self,
                 compare_lists,
                 input_index,
                 source_perspective_id,
                 perspective_name_list,
                 storage,
                 host_url,
                 cache_kwargs,
                 four_tensors=False,
                 truth_threshold=0.97,
                 only_orphans_flag=True):

        self.compare_lists = compare_lists
        self.input_index = input_index
        self.source_perspective_id = source_perspective_id
        self.four_tensors = four_tensors
        self.truth_threshold = truth_threshold
        self.perspective_name_list = perspective_name_list
        self.storage = storage
        self.host_url = host_url
        self.cache_kwargs = cache_kwargs
        self.only_orphans_flag = only_orphans_flag

        project_dir = os.path.abspath(os.getcwd())
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        os.chdir(script_dir)

        # Загрузка конфигурации
        try:
            with open('config99.json', 'r', encoding='utf-8') as config_file:
                config = json.load(config_file)
        except (FileNotFoundError, json.JSONDecodeError):
            config = {}
            print("Используются настройки по умолчанию")

        self.max_len = config.get('sequence_length', 100)

        # Загрузка токенизатора
        with open('tokenizer.json', 'r', encoding='utf-8') as f:
            self.tokenizer = tokenizer_from_json(json.load(f))

        # Загрузка моделей
        if four_tensors:
            self.model_distilled = tf.keras.models.load_model(  # Новая модель
                'final2.keras',
                custom_objects={'AbsDiffLayer': AbsDiffLayer}
            )
        else:
            self.model_dict = tf.keras.models.load_model(
                'fasttext_model.keras',
                custom_objects={'AbsDiffLayer': AbsDiffLayer}
            )

        # Change dir back
        os.chdir(project_dir)

    @celery.task
    def predict_cognates(
            self,
            word_pairs,
            task):

        def split_items(items, input_links=None):
            links = 0
            result = ([], [], [], [])

            for i in range(len(items)):
                if input_links and set(input_links) & set(items[i][3]):
                    links += 1
                    continue

                for j in range(4):
                    result[j].append(items[i][j])

            return result, links

        def process_text(text):
            chars = list(text.lower())
            seq = self.tokenizer.texts_to_sequences([chars])[0]
            return pad_sequences([seq], maxlen=self.max_len, padding='post', truncating='post')[0]

        stamp_file = os.path.join(self.storage['path'], 'lingvodoc_stamps', str(task.id))

        # Calculate prediction
        def get_prediction(input_word, input_tran, input_id, input_links, X_word, X_tran, event):

            if event.is_set():
                return None

            count = 0
            links_total = 0
            similarities = []

            # Проход по каждому списку для сравнения
            for i, compare_list in enumerate(self.compare_lists):

                if not compare_list:
                    continue

                (compare_words, compare_trans, compare_lex_ids, _), links = (
                    split_items(compare_list, input_links))

                links_total += links
                X_compare_words = [process_text(w) for w in compare_words]
                X_compare_trans = [process_text(t) for t in compare_trans]

                if not self.four_tensors:

                    if os.path.isfile(stamp_file):
                        event.set()
                        return None

                    inputs = [
                        np.array([X_word] * len(compare_words)),
                        np.array(X_compare_words),
                        np.array([X_tran] * len(compare_words)),
                        np.array(X_compare_trans)
                    ]

                    predictions = self.model_dict.predict(inputs).flatten()

                    for compare_word, compare_tran, compare_id, pred in zip(
                            compare_words, compare_trans, compare_lex_ids, predictions):

                        if pred > self.truth_threshold:
                            similarities.append((i, [compare_word, compare_tran], compare_id, f"{pred:.4f}"))

                    continue

                for compare_word, compare_tran, compare_id, X_comp_word, X_comp_tran in zip(
                        compare_words, compare_trans, compare_lex_ids, X_compare_words, X_compare_trans):

                    # Checking stamp-to-stop every hundred comparings
                    if count % 100 == 0 and os.path.isfile(stamp_file):
                        event.set()
                        return None

                    count += 1

                    inputs = [
                        X_word,
                        X_tran,
                        X_comp_word,
                        X_comp_tran,
                    ]

                    pred = self.model_distilled.predict(inputs)[0][0]

                    if pred > self.truth_threshold:
                        similarities.append((i, [compare_word, compare_tran], compare_id, f"{pred:.4f}"))

            if os.path.isfile(stamp_file):
                event.set()
                return None

            return (
                [(
                    self.input_index,
                    f"{input_word} '{input_tran}'",
                    input_id,
                    None,
                    similarities,
                    []
                )] if len(similarities) else []
            ), links_total

        start_time = now()
        results = []
        group_count = 0
        current_stage = 0
        flushed = 0
        result_link = ""
        input_len = len(word_pairs)
        compare_len = sum(map(len, self.compare_lists))
        initialize_cache(self.cache_kwargs)
        task = TaskStatus.get_from_cache(task.key)

        def add_result(res):

            nonlocal current_stage, flushed, result_link, group_count

            if res is None:
                return

            result, links = res

            results.extend(result)
            group_count += links

            current_stage += 1
            finished = (current_stage == input_len)
            passed = now() - start_time
            left = passed / current_stage * input_len - passed

            days = int(left / 86400)
            hours = int((left - days * 86400) / 3600)
            minutes = int((left - days * 86400 - hours * 3600) / 60)

            progress = 100 if finished else int(current_stage / input_len * 100)
            status = "Finished" if finished else f"~ {days}d:{hours}h:{minutes}m left ~"

            if passed - flushed > 300 or finished:
                flushed = passed

                result_dict = (
                    dict(
                        suggestion_list=results,
                        perspective_name_list=self.perspective_name_list,
                        transcription_count=compare_len * current_stage,
                        group_count=f"{group_count} filtered" if self.only_orphans_flag else "non-filtered",
                        source_perspective_id=self.source_perspective_id))

                storage_dir = os.path.join(self.storage['path'], 'neuro_cognates')
                pickle_path = os.path.join(storage_dir, str(task.id))
                os.makedirs(storage_dir, exist_ok=True)

                with gzip.open(pickle_path, 'wb') as result_data_file:
                    pickle.dump(result_dict, result_data_file)

                result_link = ''.join([self.host_url, '/suggestions/', str(task.id)])

            task.set(current_stage, progress, status, result_link)

        with multiprocess.Pool(multiprocess.cpu_count() // 2) as p:

            # Разделяем входные пары на слова и переводы
            (input_words, input_trans, input_lex_ids, input_linked_groups), _ = split_items(word_pairs)

            # Токенизация и паддинг входных данных
            X_input_words = [process_text(w) for w in input_words]
            X_input_trans = [process_text(t) for t in input_trans]

            event = multiprocess.Manager().Event()

            task.set(1, 0, "first words processing...")

            for args in zip(
                    input_words,
                    input_trans,
                    input_lex_ids,
                    input_linked_groups,
                    X_input_words,
                    X_input_trans,
                    [event] * input_len
            ):
                p.apply_async(get_prediction, args, callback=add_result,
                              error_callback=(lambda e: print(e, flush=True)))

            p.close()

            # Terminate all the processes on event
            event.wait()
            print("Killed process !!!")
            task.set(None, -1, "Stopped manually", result_link)
            p.terminate()

        # Removing stamp-to-stop if exists
        try:
            os.remove(stamp_file)
        except OSError:
            pass

        return results

    def index(self, word_pairs, task):

        return NeuroCognates.predict_cognates.delay(
            self,
            word_pairs,
            task)
