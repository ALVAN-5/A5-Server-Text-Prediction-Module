# A5-Server-Text-Prediction-Module
Module to predict which action a query best fits

## To Install

```bash
pip install alvan-text-predictor
```

## To Use
```python
from prediction import predictor as pd

predictor = pd.Predictor('intents.json')

predictor.query('turn the lights off')
```

### Predictor(intents_filename: str, pickle_file: str="data.pickle", intents_cache_file: str="intents_cache") -> Predictor

The constructor for the Predictor class takes in 1 required and 2 optional parameters: intents_filename, pickle_file, and intents_cache_file

| Parameter           | Type | Required | Default  | Description |
| --------------------|-------|-----| ----------- |------------|
| intents_filename    | `str` | &#9745; | `""` | Filename for the intents file, used to train the model.       |
| pickle_file         | `str` | &#9744; | `"data.pickle"` | Filename for the file that contains/will contain the serialized model-training output        |
| intents_cache_file  | `str` | &#9744; | `"intents_cache"` | Filename for the intents cache. This is used to detect if any changes have been made to the original intents file. If the contents of this file differ from the file passed into the `intents_filename`, the model will be retrained and the contents of this file and the `pickle_file` will be overwritten with the contents of the `intents_filename` file, and the serialized output of the training.        |

### query(query)

The query method takes in a query, compares it to the trained intents, and replies with an object indicating which intent it most closely matches.

The output has the following shape:
```python
{intent responses, intent flags}
```
Example
```python
predictor.query('Turn off the lights') -> {4, {"followup": 0}}
```
_based on example below_


### Intents File
The intents file is a json file with the following shape:
```json
{"intents": [
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
```
Example:
```python
{"intents": [
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
        "flags": {"followup": 0}
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
        "flags": {"followup": 0}
    },
]}
```

## Corequisites

nltk must be installed alongside.
```bash
pip install nltk
```
