# A5-Server-Text-Prediction-Module
Module to predict which action a query best fits

## To Install

```bash
pip install -e git://github.com/ALVAN-5/A5-Server-Text-Prediction-Module.git@{version number}#egg={ desired egg name }
```

## To Use
```python
from prediction import predictor as pd

predictor = pd.Predictor('intents.json')

predictor.query('turn the lights off')
```