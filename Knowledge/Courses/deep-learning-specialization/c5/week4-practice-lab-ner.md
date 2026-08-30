Packages

```python
import pandas as pd
import numpy as np
import tensorflow as tf
import json
import random
import logging
import re

tf.get_logger().setLevel('ERROR')
```

Named-Entity Recognition to Process Resumes

- when faced with a large amount of unstructured text data, named-entity recognition (NER) can help detect and classify important information in the dataset
- for instance, in the running example “Jane visits Africa in September”, NER would help detect “Jane”, “Africa”, and “September” as named-entities and classify them as person, location, and time
    - use a variation of the transformer model to process a large dataset of resumes
    - find and classify relevant information such as the companies the applicant worked at, skills, type of degree, etc.
- Dataset Cleaning
    - optimize a transformer model on a dataset of resumes

        ```python
        df_data = pd.read_json("ner.json", lines=True)
        df_data = df_data.drop(['extras'], axis=1)
        df_data['content'] = df_data['content'].str.replace("\n", " ")
        ```

        ```python
        df_data.head()
        ```

        ```python
        df_data.iloc[0]['annotation']

        '''
        [{'label': ['Skills'],
          'points': [{'start': 1295,
            'end': 1621,
            'text': '\n• Programming language: C, C++, Java\n• Oracle PeopleSoft\n• Internet Of Things\n• Machine Learning\n• Database Management System\n• Computer Networks\n• Operating System worked on: Linux, Windows, Mac\n\nNon - Technical Skills\n\n• Honest and Hard-Working\n• Tolerant and Flexible to Different Situations\n• Polite and Calm\n• Team-Player'}]},
         {'label': ['Skills'],
          'points': [{'start': 993,
            'end': 1153,
            'text': 'C (Less than 1 year), Database (Less than 1 year), Database Management (Less than 1 year),\nDatabase Management System (Less than 1 year), Java (Less than 1 year)'}]},
        ...
         {'label': ['Companies worked at'],
          'points': [{'start': 49, 'end': 57, 'text': 'Accenture'}]},
         {'label': ['Designation'],
          'points': [{'start': 13,
            'end': 45,
            'text': 'Application Development Associate'}]},
         {'label': ['Name'],
          'points': [{'start': 0, 'end': 11, 'text': 'Abhishek Jha'}]}]
        '''
        ```

        ```python
        def mergeIntervals(intervals):
            sorted_by_lower_bound = sorted(intervals, key=lambda tup: tup[0])
            merged = []

            for higher in sorted_by_lower_bound:
                if not merged:
                    merged.append(higher)
                else:
                    lower = merged[-1]
                    if higher[0] <= lower[1]:
                        if lower[2] is higher[2]:
                            upper_bound = max(lower[1], higher[1])
                            merged[-1] = (lower[0], upper_bound, lower[2])
                        else:
                            if lower[1] > higher[1]:
                                merged[-1] = lower
                            else:
                                merged[-1] = (lower[0], higher[1], higher[2])
                    else:
                        merged.append(higher)
            return merged
        ```

        ```python
        def get_entities(df):

            entities = []

            for i in range(len(df)):
                entity = []

                for annot in df['annotation'][i]:
                    try:
                        ent = annot['label'][0]
                        start = annot['points'][0]['start']
                        end = annot['points'][0]['end'] + 1
                        entity.append((start, end, ent))
                    except:
                        pass

                entity = mergeIntervals(entity)
                entities.append(entity)

            return entities
        ```

        ```python
        df_data['entities'] = get_entities(df_data)
        df_data.head()
        ```

        ```python
        def convert_dataturks_to_spacy(dataturks_JSON_FilePath):
            try:
                training_data = []
                lines=[]
                with open(dataturks_JSON_FilePath, 'r') as f:
                    lines = f.readlines()

                for line in lines:
                    data = json.loads(line)
                    text = data['content'].replace("\n", " ")
                    entities = []
                    data_annotations = data['annotation']
                    if data_annotations is not None:
                        for annotation in data_annotations:
                            #only a single point in text annotation.
                            point = annotation['points'][0]
                            labels = annotation['label']
                            # handle both list of labels or a single label.
                            if not isinstance(labels, list):
                                labels = [labels]

                            for label in labels:
                                point_start = point['start']
                                point_end = point['end']
                                point_text = point['text']

                                lstrip_diff = len(point_text) - len(point_text.lstrip())
                                rstrip_diff = len(point_text) - len(point_text.rstrip())
                                if lstrip_diff != 0:
                                    point_start = point_start + lstrip_diff
                                if rstrip_diff != 0:
                                    point_end = point_end - rstrip_diff
                                entities.append((point_start, point_end + 1 , label))
                    training_data.append((text, {"entities" : entities}))
                return training_data
            except Exception as e:
                logging.exception("Unable to process " + dataturks_JSON_FilePath + "\n" + "error = " + str(e))
                return None

        def trim_entity_spans(data: list) -> list:
            """Removes leading and trailing white spaces from entity spans.

            Args:
                data (list): The data to be cleaned in spaCy JSON format.

            Returns:
                list: The cleaned data.
            """
            invalid_span_tokens = re.compile(r'\s')

            cleaned_data = []
            for text, annotations in data:
                entities = annotations['entities']
                valid_entities = []
                for start, end, label in entities:
                    valid_start = start
                    valid_end = end
                    while valid_start < len(text) and invalid_span_tokens.match(
                            text[valid_start]):
                        valid_start += 1
                    while valid_end > 1 and invalid_span_tokens.match(
                            text[valid_end - 1]):
                        valid_end -= 1
                    valid_entities.append([valid_start, valid_end, label])
                cleaned_data.append([text, {'entities': valid_entities}])
            return cleaned_data  
        ```

        ```python
        data = trim_entity_spans(convert_dataturks_to_spacy("ner.json"))
        ```

        ```python
        from tqdm.notebook import tqdm
        def clean_dataset(data):
            cleanedDF = pd.DataFrame(columns=["sentences_cleaned"])
            sum1 = 0
            for i in tqdm(range(len(data))):
                start = 0
                emptyList = ["Empty"] * len(data[i][0].split())
                numberOfWords = 0
                lenOfString = len(data[i][0])
                strData = data[i][0]
                strDictData = data[i][1]
                lastIndexOfSpace = strData.rfind(' ')
                for i in range(lenOfString):
                    if (strData[i]==" " and strData[i+1]!=" "):
                        for k,v in strDictData.items():
                            for j in range(len(v)):
                                entList = v[len(v)-j-1]
                                if (start>=int(entList[0]) and i<=int(entList[1])):
                                    emptyList[numberOfWords] = entList[2]
                                    break
                                else:
                                    continue
                        start = i + 1  
                        numberOfWords += 1
                    if (i == lastIndexOfSpace):
                        for j in range(len(v)):
                                entList = v[len(v)-j-1]
                                if (lastIndexOfSpace>=int(entList[0]) and lenOfString<=int(entList[1])):
                                    emptyList[numberOfWords] = entList[2]
                                    numberOfWords += 1
                cleanedDF = cleanedDF.append(pd.Series([emptyList],  index=cleanedDF.columns ), ignore_index=True )
                sum1 = sum1 + numberOfWords
            return cleanedDF
        ```

        ```python
        cleanedDF = clean_dataset(data)
        ```

    - look at the cleaned dataset and the categories the named-entities are matched to

        ```python
        cleanedDF.head()
        ```

- Padding and Generating Tags
    - time to generate a list of unique tags to match the named-entities to

        ```python
        unique_tags = set(cleanedDF['setences_cleaned'].explode().unique())#pd.unique(cleanedDF['setences_cleaned'])#set(tag for doc in cleanedDF['setences_cleaned'].values.tolist() for tag in doc)
        tag2id = {tag: id for id, tag in enumerate(unique_tags)}
        id2tag = {id: tag for tag, id in tag2id.items()}
        ```

        ```python
        unique_tags

        '''
        {'College Name',
         'Companies worked at',
         'Degree',
         'Designation',
         'Email Address',
         'Empty',
         'Graduation Year',
         'Location',
         'Name',
         'Skills',
         'UNKNOWN',
         'Years of Experience'}
        '''
        ```

    - create an array of tags from the cleaned dataset
    - oftentimes, the input sequence can exceed the max length of a sequence the network can process, so it needs to be cut off to that desired max length
    - and when the input sequence is shorter than the desired length, append zeros onto its end using Keras padding API

        ```python
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        ```

        ```python
        MAX_LEN = 512
        labels = cleanedDF['sentences_cleaned'].values.tolist()

        tags = pad_sequences([[tag2id.get(l) for l in lab] for lab in labels],
                             maxlen=MAX_LEN, value=tag2id["Empty"], padding="post",
                             dtype="long", truncating="post")
        ```

        ```python
        tags

        '''
        array([[ 7,  7,  6, ..., 11, 11, 11],
               [ 7,  7, 11, ..., 11, 11, 11],
               [ 7,  7,  7, ..., 11,  2, 11],
               ...,
               [ 7,  7,  6, ..., 11, 11, 11],
               [ 7,  7,  6, ..., 11, 11, 11],
               [ 7,  7,  6, ..., 11, 11, 11]])
        '''
        ```

- Tokenize and Align Labels with Hugging Face Library
    - before feeding the texts to transformer model, tokenize the input using a Hugging Face transformer tokenizer
    - it is crucial that the tokenizer matches the transformer model type
    - use the DistilBERT fast tokenizer, which standardizes the length of the sequence to 512 and pads with zeros
    - notice this matches the max length used when creating tags’

        ```python
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_virtual_device_configuration(gpu,[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
        ```

        ```python
        from transformers import DistilBertTokenizerFast #, TFDistilBertModel
        tokenizer = DistilBertTokenizerFast.from_pretrained('tokenizer/')
        ```

    - transformer models are often trained by tokenizers that split words into subwords
    - for instance, the word ‘Africa’ might get split into multiple subtokens
    - this can create some misalignment between the list of tags for the dataset and the list of labels generated by the tokenizer, since the tokenizer can split one word into several, or add special tokens
    - before processing, it’s important to align the list of tags and the list of labels generated by the selected tokenizer with a tokenize_and_align_labels() function
    - Exercise 1 - tokenize_and_align_labels
        - the tokenizer cuts sequences that exceed the max size allowed by the model with the param truncation=True
        - aligns the list of tags and labels with the tokenizer words_ids method returns a list that maps the subtokens to the original word in the sentence and special tokens to None
        - set the labels of all the special tokens (None) to -100 to prevent them from affecting the loss function
        - label of the first subtoken of a word and set the label for the following subtokens to -100

            ```python
            label_all_tokens = True
            def tokenize_and_align_labels(tokenizer, examples, tags):
                tokenized_inputs = tokenizer(examples, truncation=True, is_split_into_words=False, padding='max_length', max_length=512)
                labels = []
                for i, label in enumerate(tags):
                    word_ids = tokenized_inputs.word_ids(batch_index=i)
                    previous_word_idx = None
                    label_ids = []
                    for word_idx in word_ids:
                        # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                        # ignored in the loss function.
                        if word_idx is None:
                            label_ids.append(-100)
                        # We set the label for the first token of each word.
                        elif word_idx != previous_word_idx:
                            label_ids.append(label[word_idx])
                        # For the other tokens in a word, we set the label to either the current label or -100, depending on
                        # the label_all_tokens flag.
                        else:
                            label_ids.append(label[word_idx] if label_all_tokens else -100)
                        previous_word_idx = word_idx

                    labels.append(label_ids)

                tokenized_inputs["labels"] = labels
                return tokenized_inputs
            ```

            ```python
            test = tokenize_and_align_labels(tokenizer, df_data['content'].values.tolist(), tags)
            train_dataset = tf.data.Dataset.from_tensor_slices((
                test['input_ids'],
                test['labels']
            ))
            ```

- Optimization
    - feed the data into a pretrained Hugging Face model
    - optimize a DistilBERT model, which matches the tokenizer used to preprocess the data

        ```python
        from transformers import TFDistilBertForTokenClassification

        model = TFDistilBertForTokenClassification.from_pretrained('model/', num_labels=len(unique_tags))
        ```

        ```python
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
        model.compile(optimizer=optimizer, loss=model.hf_compute_loss, metrics=['accuracy']) # can also use any keras loss fn
        model.fit(train_dataset.batch(4),
                  epochs=10, 
                  batch_size=4)

        '''
        Epoch 1/10
        55/55 [==============================] - 12s 105ms/step - loss: 0.8264 - accuracy: 0.7097
        Epoch 2/10
        55/55 [==============================] - 6s 104ms/step - loss: 0.4668 - accuracy: 0.7542
        ...
        Epoch 9/10
        55/55 [==============================] - 6s 104ms/step - loss: 0.3398 - accuracy: 0.7667
        Epoch 10/10
        55/55 [==============================] - 6s 104ms/step - loss: 0.3235 - accuracy: 0.7700
        '''
        ```

        ```python
        text = "Manisha Bharti. 3.5 years of professional IT experience in Banking and Finance domain"
        inputs = tokenizer(text, return_tensors="tf", truncation=True, is_split_into_words=False, padding="max_length", max_length=512 )
        input_ids = inputs["input_ids"]
        #inputs["labels"] = tf.reshape(tf.constant([1] * tf.size(input_ids).numpy()), (-1, tf.size(input_ids)))
        ```

        ```python
        output = model(inputs).logits
        prediction = np.argmax(output, axis=2)
        print(prediction)

        '''
        [[8 2 2 2 2 2 8 8 8 8 8 8 8 8 8 8 8 8 8 8 5 8 8 8 8 8 2 8 8 8 8 8 8 8 8 8
          8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 2 2 2 2 8 8 8 8 8 8 8 8 8 8 8 8
          8 8 8 8 8 8 8 8 8 8 8 8 8 8 2 2 2 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8
          8 8 8 2 2 2 2 2 8 8 8 8 8 8 8 8 8 8 8 8 8 8 2 2 2 2 8 8 8 8 8 8 8 8 8 8
          8 2 2 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 2 2 8 8 2 2 2 2
          8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 2 8 8 8 8 2 2 2 2 2
          8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 2 8 8 8 8 8 8 8 8 8 8
          8 2 2 2 2 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8
          8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8
          8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 2 2 2 2 8 8 8 8 8 8 8 8 8 8 8 8
          8 8 8 8 8 8 8 8 8 8 2 2 2 2 8 8 2 8 8 8 8 8 8 8 2 2 2 2 8 2 2 2 2 8 8 2
          2 2 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 2 2 2 2 2 2 2 8 8 8 8 8 8
          8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 2 8 8 8 2 2 2 2 2 8 8 8 8 8 8 8
          8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 2 8 8 8 8 8 8 8 8 8 8
          8 8 8 8 8 8 8 8]]
        '''
        ```

        ```python
        model(inputs)

        '''
        model(inputs)

        TFTokenClassifierOutput(loss=None, logits=<tf.Tensor: shape=(1, 512, 12), dtype=float32, numpy=
        array([[[-0.5268031 , -0.41972262,  0.60984033, ..., -0.10265747,
                  0.3157095 , -0.42518455],
                [-0.35176107, -0.5857346 ,  2.645111  , ..., -0.43226254,
                 -0.29312125, -0.2880653 ],
                [-0.47827202, -0.80437434,  2.7233071 , ..., -0.5174456 ,
                 -0.22817095, -0.4205181 ],
                ...,
                [-0.62451065, -0.6474978 , -0.26765507, ..., -0.05570607,
                  0.06908587, -0.18547551],
                [-0.54089874, -0.53435534, -0.23902452, ..., -0.04666487,
                  0.09474991, -0.1920171 ],
                [-0.64895296, -0.61249495, -0.19486724, ..., -0.14808854,
                  0.07140078, -0.10775104]]], dtype=float32)>, hidden_states=None, attentions=None)
        '''
        ```

        ```python
        pred_labels = []

        !pip install seqeval
        ```

        ```python
        true_labels = [[id2tag.get(true_index, "Empty") for true_index in test['labels'][i]] for i in range(len(test['labels']))]
        np.array(true_labels).shape

        # (220, 512)
        ```

        ```python
        output = model.predict(train_dataset)

        # 220/220 [==============================] - 2s 5ms/step
        ```

        ```python
        predictions = np.argmax(output['logits'].reshape(220, -1, 12), axis=-1)

        predictions.shape

        # (220, 512)
        ```

        ```python
        from matplotlib import pyplot as plt 

        p = plt.hist(np.array(true_labels).flatten())
        plt.xticks(rotation='vertical')
        plt.show()
        ```

        ```python
        from collections import Counter
        Counter(np.array(true_labels).flatten())

        '''
        Counter({'Empty': 103155,
                 'Name': 1035,
                 'Designation': 1100,
                 'Email Address': 76,
                 'Companies worked at': 908,
                 'College Name': 792,
                 'Skills': 4704,
                 'Degree': 598,
                 'Graduation Year': 80,
                 'Location': 116,
                 'Years of Experience': 75,
                 'UNKNOWN': 1})
        '''
        ```

        ```python
        pred_labels = [[id2tag.get(index, "Empty") for index in predictions[i]] for i in range(len(predictions))]
        p = plt.hist(np.array(pred_labels).flatten())
        plt.xticks(rotation='vertical')
        plt.show()
        ```

        ```python
        from seqeval.metrics import classification_report
        print(classification_report(true_labels, pred_labels))

        '''
                            precision    recall  f1-score   support

                    NKNOWN       0.00      0.00      0.00         1
                       ame       0.89      0.91      0.90       220
        ears of Experience       0.00      0.00      0.00        37
                     egree       0.00      0.00      0.00       144
                esignation       0.40      0.23      0.29       430
                     kills       0.41      0.51      0.46      4704
              mail Address       0.00      0.00      0.00        76
                      mpty       0.95      0.97      0.96    103155
                   ocation       0.00      0.00      0.00        73
               ollege Name       0.00      0.00      0.00       214
        ompanies worked at       0.00      0.00      0.00       470
            raduation Year       0.00      0.00      0.00        58

                 micro avg       0.92      0.93      0.93    109582
                 macro avg       0.22      0.22      0.22    109582
              weighted avg       0.91      0.93      0.92    109582
        '''
        ```