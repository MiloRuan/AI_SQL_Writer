from nltk.corpus import brown
from nltk.corpus import conll2000
from nltk.classify.maxent import MaxentClassifier
import nltk
from nltk.tag.sequential import ClassifierBasedPOSTagger
from nltk.tag import UnigramTagger, BigramTagger, TrigramTagger, SequentialBackoffTagger


def backoff_tagger(train_sents, tagger_classes, backoff=None):
    for cls in tagger_classes:
        backoff = cls(train_sents, backoff=backoff)
    return backoff


class SQLPosTagger(SequentialBackoffTagger):
    def isEntity(self, word):
        entities = ['DW_F_JOURNAL','DW_F_JOURNAL.BUSINESS_DATE','DW_F_JOURNAL.CURRENCY','DW_F_JOURNAL.FTP_ALN','DW_F_JOURNAL.ID_ENTITY','DW_F_JOURNAL.SITE_CODE','DW_F_LEDGER_BALANCE','DW_F_LEDGER_BALANCE.ALN','DW_F_LEDGER_BALANCE.BUSINESS_DATE','DW_F_LEDGER_BALANCE.CCY','DW_F_LEDGER_BALANCE.ID_ENTITY','DW_F_LEDGER_BALANCE.SITE_CODE','DW_H_BUSINESS_NATURE','DW_H_BUSINESS_NATURE.','FINANCIAL_ELEMENT']
        if word in entities:
            return True
        else:
            return False


    def isVerb(self, word):
        verbList = ['search', 'return', 'query', 'get']
        if word.lower() in verbList:
            return True
        else:
            return False


    def isAdv(self, word):
        advList = ['right', 'left', 'inner', 'outter', 'full']
        if word.lower() in advList:
            return True
        else:
            return False

    def choose_tag(self, tokens, index, history):
        if index == 0 and SQLPosTagger.isVerb(self, tokens[index]):
            return "VB"
        if SQLPosTagger.isAdv(self, tokens[index]) and tokens[index + 1] == "join" :
            return "ADJ"
        if SQLPosTagger.isEntity(self, tokens[index]):
            return "NN"
        return None


def myParse(sentence):
    print("ClassifierBasedPOSTagger tag:")
    brown_tagged_sents = brown.tagged_sents(categories='news')
    train_sents = brown_tagged_sents[:500000]
    tagger = ClassifierBasedPOSTagger(train=train_sents)  # , classifier_builder=MaxentClassifier.train)
    mytagger = SQLPosTagger(tagger)

    words = nltk.word_tokenize(sentence)
    result = mytagger.tag(words)
    print(result)

def defaultParse(sentence):
    print("nltk default tag:")
    tokens = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)
    print(tagged)

if __name__ == '__main__':
    #sentence = "Search the data from database"
    sentence = "DW_F_JOURNAL outter join DW_F_JOURNAL)"
    myParse(sentence)
    defaultParse(sentence)
