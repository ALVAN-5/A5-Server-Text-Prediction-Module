from prediction import predictor as pd

def test_intents():
  predictor = pd.Predictor('tests/test_data/test_intents1.json')
  res = predictor.query('turn the lights on')
  assert res == (4, {'1': 4})

def test_new_training_data():
  predictor = pd.Predictor('tests/test_data/test_intents1.json')
  res = predictor.query('turn the lights on')
  assert res == (4, {'1': 4})
  predictor = pd.Predictor('tests/test_data/test_intents2.json')
  res = predictor.query('turn the lights on')
  assert res == (4, {'1': 3})
