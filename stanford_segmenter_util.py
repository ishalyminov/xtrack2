import os
import shutil
import requests
from zipfile import ZipFile
from StringIO import StringIO

from nltk.tokenize.stanford_segmenter import StanfordSegmenter


STANFORD_SEGMENTER_URL = \
    'http://nlp.stanford.edu/software/stanford-segmenter-2015-12-09.zip'
STANFORD_SEGMENTER_DIR = \
    os.path.splitext(STANFORD_SEGMENTER_URL.split('/')[-1])[0]
STANFORD_SEGMENTER_JAR = os.path.join(
    STANFORD_SEGMENTER_DIR,
    'stanford-segmenter-3.6.0.jar'
)
STANFORD_SIHAN_CORPORA_DICT = os.path.join(STANFORD_SEGMENTER_DIR, 'data')
STANFORD_MODEL = os.path.join(STANFORD_SEGMENTER_DIR, 'data', 'pku.gz')
STANFORD_DICT = os.path.join(
    STANFORD_SEGMENTER_DIR,
    'data',
    'dict-chris6.ser.gz'
)

os.environ['CLASSPATH'] = os.path.join(
    os.path.dirname(__file__),
    STANFORD_SEGMENTER_DIR,
    'slf4j-api.jar'
)

STANFORD_SEGMENTER = None


def download_stanford_segmenter():
    if os.path.isdir(STANFORD_SEGMENTER_DIR):
        shutil.rmtree(STANFORD_SEGMENTER_DIR)
    stanford_response = requests.get(STANFORD_SEGMENTER_URL)
    if not stanford_response.ok:
        raise RuntimeError('Stanford segmenter unavailable')
    archive = stanford_response.content
    archive_zip = ZipFile(StringIO(archive))
    archive_zip.extractall()


def get_stanford_segmenter():
    if not os.path.isdir(STANFORD_SEGMENTER_DIR):
        download_stanford_segmenter()
    global STANFORD_SEGMENTER
    if not STANFORD_SEGMENTER:
        STANFORD_SEGMENTER = StanfordSegmenter(
            path_to_jar=STANFORD_SEGMENTER_JAR,
            path_to_sihan_corpora_dict=STANFORD_SIHAN_CORPORA_DICT,
            path_to_model=STANFORD_MODEL,
            path_to_dict=STANFORD_DICT,
            verbose=True
        )
    return STANFORD_SEGMENTER


def segment(in_text, list_output=True):
    result = get_stanford_segmenter().segment(in_text).strip()
    if list_output:
        result = result.split()
    return result
