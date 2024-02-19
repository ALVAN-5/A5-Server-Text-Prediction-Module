import os
from prediction import predictor as pd
import pytest

SKIP_DELETE = "SKIP_DELETE"

pickle_file = ""


def test_intents():
    global pickle_file
    pickle_file = "test_intents.pickle"
    predictor = pd.Predictor(os.path.join('tests', 'test_data', 'test_intents1.json'), pickle_file=pickle_file)
    res = predictor.query('turn the lights on')
    assert res == ('lights_on', '', 4, {'1': 4}) 


def test_new_training_data():
    global pickle_file
    pickle_file = "test_new_training_data.pickle"
    predictor = pd.Predictor(os.path.join('tests', 'test_data', 'test_intents1.json'), pickle_file=pickle_file)
    res = predictor.query('turn the lights on')
    assert res == ('lights_on', '', 4, {'1': 4}) 
    predictor = pd.Predictor(os.path.join('tests', 'test_data', 'test_intents2.json'), pickle_file=pickle_file)
    res = predictor.query('turn the lights on')
    assert res == ('lights_on', '', 4, {'1': 3}) 


def test_no_pickle_file_provided():
    global pickle_file
    pickle_file = SKIP_DELETE
    predictor = pd.Predictor(os.path.join('tests', 'test_data', 'test_intents1.json'))
    assert predictor.pickle_file == "data.pickle"


def test_no_intents_cache_file_provided():
    global pickle_file
    pickle_file = SKIP_DELETE
    predictor = pd.Predictor(os.path.join('tests', 'test_data', 'test_intents1.json'))
    assert predictor.intents_cache_file == "intents_cache"


def test_intents_file_does_not_exist():
    global pickle_file
    pickle_file = SKIP_DELETE
    assert os.path.isfile('this_file_does_not_exist.json') is False
    with pytest.raises(FileNotFoundError):
        pd.Predictor('this_file_does_not_exist.json')


def test_intents_cache_file_does_not_exist():
    global pickle_file
    pickle_file = SKIP_DELETE
    filename = 'this_file_does_not_exist'
    full_path = os.path.join('.PredictorCache', filename)
    try:
        os.remove(full_path)
    except FileNotFoundError:
        pass

    assert os.path.isfile(full_path) is False
    pd.Predictor(os.path.join('tests', 'test_data', 'test_intents1.json'), intents_cache_file=filename)
    assert os.path.isfile(full_path) is True
    os.remove(full_path)


@pytest.fixture(autouse=True)
def run_before_and_after_tests(tmpdir):
    """Fixture to execute asserts before and after a test is run"""
    global pickle_file
    # Setup: fill with any logic you want

    yield  # this is where the testing happens

    # Teardown : fill with any logic you want
    delete_pickle_file(pickle_file)


def delete_pickle_file(pickle_file):
    if pickle_file != SKIP_DELETE:
        os.remove(os.path.join(".PredictorCache", pickle_file))
