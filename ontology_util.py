import sys
import os
from dstc5_scripts import ontology_reader

ONTOLOGY_FILE = os.path.join('dstc5_scripts', 'config', 'ontology_dstc5.json')
ONTOLOGY = ontology_reader.OntologyReader(ONTOLOGY_FILE)


def get_topics():
    return ' '.join(ONTOLOGY.get_topics())


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Usage: ontology_util.py <command>'
        exit()
    command = sys.argv[1]
    print locals()[command]()
