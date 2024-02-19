import json
import nltk
import os
import pickle
import typing


class Predictor:
    '''
    Module for determining which predefined message the provided query is
    asking.
    '''
    def __init__(
        self,
        intents_filename: str,
        pickle_file: str = "data.pickle",
        intents_cache_file: str = "intents_cache"
    ):
        '''
        Takes in JSON filename of predefined messages the user could be asking, filename for pickled data, and a
            filename for the intents cache.

        pickle_file defaults to "data.pickle" and is used to save the trained model to a serialized object
        intents_cache_file defaults to "intents_cache" and is used to detect any changes between the intents file and
            whatever intents file was previously used for training the.

        If the contents of the intents file and the intents cache file are the same, the model will load the trained
            from the pickle_file; otherwise, the model will be trained on the contents of the intents file, copy that
            content to the intents_cache_file, and serialize the trained output to the data.pickle file.

        The json hould be in the following format
        {"intents: [
            <List of Objects representing intents with the following shape:
            {
                "tag":<String - human-readable descriptor>,
                "patterns": [<List of Strings - various ways a user could phrase their query],
                "responses": <Int - unique id of the intent>,
                "context_set":<String - currently unused metadata>,
                "flags": <Object with boolean values for flags used by the consumer>
            }
            >
        ]}

        Example:
        {"intents: [
            {
                "tag":"lights_on",
                "patterns": [
                    "turn on the lights",
                    "it is too dark in here",
                    "I want the lights on",
                    "I want it brighter in here"
                ],
                "responses":4,
                "context_set":"",
                "flags": {"1": 4}
            },
            {
                "tag":"lights_off",
                "patterns": [
                    "turn off the lights",
                    "it is too bright in here",
                    "I want the lights off",
                    "I want it darker in here"
                ],
                "responses":5,
                "context_set":"",
                "flags": false
            },
        ]}

        '''
        os.makedirs(".PredictorCache", exist_ok=True)
        self.pickle_file = pickle_file
        self.intents_cache_file = intents_cache_file

        self.stemmer = nltk.stem.LancasterStemmer()

        try:
            with open(os.path.join(".PredictorCache", self.intents_cache_file), "r") as cached_intents_file:
                cached_intents = cached_intents_file.read()
        except FileNotFoundError:
            cached_intents = ""

        with open(intents_filename, "r") as intents_file:
            intents = intents_file.read()

        with open(intents_filename, "rb") as intents_file_bytes:
            self.intents_json = json.load(intents_file_bytes)

        open_pickled_data = False
        try:
            with open(os.path.join(".PredictorCache", self.pickle_file), "rb") as file_bytes:
                self.words, self.labels, self.training, self.output = pickle.load(file_bytes)

            open_pickled_data = True
        except FileNotFoundError:
            pass

        if cached_intents != intents or not open_pickled_data:
            with open(intents_filename, "r") as intents_file:
                self.__train_model_from_intents(json.load(intents_file))

            with open(os.path.join(".PredictorCache", self.intents_cache_file), "w") as ic:
                ic.write(intents)

    def __train_model_from_intents(self, intents_json: typing.Dict[str, list[typing.Dict[str, str]]]):
        '''
        Private method to train data on new intents JSON file
        Takes in result of json.load(<Intents JSON file>)
        '''
        print("Training data")
        self.intents_json = intents_json
        words = []
        labels = []
        docs_x = []
        docs_y = []

        for intent in self.intents_json["intents"]:
            for pattern in intent["patterns"]:
                wrds = nltk.word_tokenize(pattern)
                words.extend(wrds)
                docs_x.append(wrds)
                docs_y.append(intent["tag"])

            if intent["tag"] not in labels:
                labels.append(intent["tag"])

        words = [self.stemmer.stem(w.lower()) for w in words if w != "?"]
        words = sorted(list(set(words)))

        labels = sorted(labels)

        training = []
        output = []

        out_empty = [0 for _ in range(len(labels))]

        for x, doc in enumerate(docs_x):
            bag = []

            wrds = [self.stemmer.stem(w.lower()) for w in doc]

            for w in words:
                if w in wrds:
                    bag.append(1)
                else:
                    bag.append(0)

            output_row = out_empty[:]
            output_row[labels.index(docs_y[x])] = 1

            training.append(bag)
            output.append(output_row)

        self.words = words
        self.labels = labels
        self.training = training
        self.output = output

        with open(os.path.join(".PredictorCache", self.pickle_file), "wb") as f:
            pickle.dump((words, labels, training, output), f)

    def query(self, query: str):
        '''
        Method to predict which intent the provided query best matches.
        Takes in a string, returns most closely matched intent, without the patterns.
        '''
        words = self.words
        training = self.training
        output = self.output

        training_lines = training
        query_words = nltk.word_tokenize(query)
        query_words = [self.stemmer.stem(w.lower()) for w in query_words if w != "?"]
        query_words = sorted(list(set(query_words)))

        lineWithMostPoints = 0
        maxPoints = -1

        for index, line in enumerate(training_lines):
            points = 0
            for wrd in query_words:
                if wrd in words and line[words.index(wrd)] == 1:
                    points += 1

            line_length = len([x for x in line if x == 1])
            if points/line_length > maxPoints:
                lineWithMostPoints = index
                maxPoints = points/line_length

        output_list = output
        final_output = []
        for output_row in output_list:
            if output_row not in final_output:
                final_output.append(output_row)

        for index, line in enumerate(final_output):
            if line == output[lineWithMostPoints]:
                return (
                    self.intents_json['intents'][index+1]['tag'],
                    self.intents_json['intents'][index+1]['context_set'],
                    self.intents_json['intents'][index+1]['responses'],
                    self.intents_json['intents'][index+1]['flags']
                )
