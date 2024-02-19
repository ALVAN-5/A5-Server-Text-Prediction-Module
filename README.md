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

## Corequisites

nltk must be installed alongside.
```bash
pip install nltk
```
