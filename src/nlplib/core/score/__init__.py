
from functools import total_ordering

from nlplib.general.math import normalize_values
from nlplib.general import literal_representation
from nlplib.core import Base

__all__ = ['WeightedFunction', 'Score', 'weighted', 'score', 'rank']

class WeightedFunction (Base) :
    ''' Weighted functions are functions that have an associated weight attribute. This allows for weighted scoring
        functions. '''

    __slots__ = ('function', 'weight', 'low_is_better')

    def __init__ (self, function, weight=1.0, low_is_better=False) :
        self.function = function

        self.weight        = float(weight)
        self.low_is_better = low_is_better

    def __repr__ (self) :
        return super().__repr__(self.function.__name__, str(hex(id(self))))

    def __call__ (self, *args, **kw) :
        return self.function(*args, **kw)

    def adjust_orientation (self, normalized_score) :
        ''' This adjusts the orientation of the score (if necessary), seeing as a low score in some metrics indicates a
            "better match." '''

        if self.low_is_better :
            return (1.0 - normalized_score)
        else :
            return normalized_score

def weighted (function, *args, **kw) :
    return WeightedFunction(function, *args, **kw)

@total_ordering
class Score (Base) :
    ''' This is an object designed to hold the scores and subscores for a particular object. '''

    __slots__ = ('object', 'score', 'subscores')

    def __init__ (self, object, score=0.0, subscores=None) :
        self.object = object

        self.score = score

        if subscores is None :
            self.subscores = {}
        else :
            self.subscores = subscores

    def __repr__ (self) :
        return literal_representation(self, self.object, score=self.score)

    def __float__ (self) :
        return float(self.score)

    def __int__ (self) :
        return int(self.score)

    def _unorderable_error (self) :
        return TypeError('Value comparisons with a <Score> object must be against an object which can be converted to '
                         'a floating point number.')

    def __lt__ (self, other) :
        try :
            return self.score < float(other)
        except (TypeError, ValueError) :
            raise self._unorderable_error()

    def __eq__ (self, other) :
        try :
            return self.score == float(other)
        except (TypeError, ValueError) :
            raise self._unorderable_error()

def _calculate_subscores (weighted_scoring_functions, object, scores) :
    for scoring_function in weighted_scoring_functions :

        unnormalized_subscores = (scoring_function(object, score.object)
                                  for score in scores)

        normalized_subscores = normalize_values(unnormalized_subscores)

        for score, normalized_subscore in zip(scores, normalized_subscores) :
            score.subscores[scoring_function] = scoring_function.adjust_orientation(normalized_subscore)

def _calculate_scores (weighted_scoring_functions, scores) :

    total_of_function_weights = sum(scoring_function.weight
                                    for scoring_function in weighted_scoring_functions)

    for score in scores :
        total_of_subscores_for_object = sum(scoring_function.weight * subscore
                                            for scoring_function, subscore in score.subscores.items())

        try :
            score.score = total_of_subscores_for_object / total_of_function_weights
        except ZeroDivisionError :
            # There were probably no scoring functions.
            score.score = 1.0

def _sort_scores (scores) :
    # todo : This probably could be done in a more efficient manner.
    sorted_by_objects_only = sorted(scores, key=lambda score : score.object)
    return sorted(sorted_by_objects_only, reverse=True)

def score (weighted_scoring_functions, object, similar_objects, sort=_sort_scores) :
    ''' This function is used to generate scores normalized between 0 and 1 (1 indicating a better match), based on a
        weighted average of multiple sub-scores. '''

    scores = [Score(similar_object) for similar_object in similar_objects]

    try :
        _calculate_subscores(weighted_scoring_functions, object, scores)
    except ValueError :
        # This happens if the <scores> list is empty.
        return scores
    else :
        _calculate_scores(weighted_scoring_functions, scores)
        return sort(scores)

def rank (*args, **kw) :
    ''' This returns objects ordered by how high their score was. '''

    for scored in score(*args, **kw) :
        yield scored.object

def __test__ (ut) :
    from nlplib.core.score.metric import distance, prevalence
    from nlplib.core.model import Word

    word = Word('the')

    words = [Word('their', prevalence=100),
             Word('platypus', prevalence=60),
             Word('there', prevalence=10),
             Word('them', prevalence=4),
             Word('th', prevalence=2),
             Word('they', prevalence=1),
             Word('though', prevalence=1)]

    def test (prevalence_weight, levenshtein_distance_weight) :

        weighted_prevalence = weighted(lambda object, similar : prevalence(similar),
                                       weight=prevalence_weight)

        weighted_levenshtein_distance = weighted(distance.levenshtein,
                                                 weight=levenshtein_distance_weight,
                                                 low_is_better=True)

        args = ([weighted_levenshtein_distance, weighted_prevalence], word, words)

        return (score(*args), rank(*args))

    # No scoring functions, means everything is a perfect match.
    ut.assert_equal(list(score([], word, words)),
                    [Score(Word('though'), score=1.0), Score(Word('they'), score=1.0), Score(Word('there'), score=1.0),
                     Score(Word('them'), score=1.0), Score(Word('their'), score=1.0), Score(Word('th'), score=1.0),
                     Score(Word('platypus'), score=1.0)])

    # Without scoring functions, this essentially just sorts the word objects alphabetically.
    ut.assert_equal(list(rank([], word, words)), sorted(words))

    scored, ranked = test(prevalence_weight=1, levenshtein_distance_weight=0)
    ut.assert_equal(list(scored),
                    [Score(Word('their'), score=1.0), Score(Word('platypus'), score=0.5959595959595959),
                     Score(Word('there'), score=0.09090909090909091), Score(Word('them'), score=0.030303030303030304),
                     Score(Word('th'), score=0.010101010101010102), Score(Word('though'), score=0.0),
                     Score(Word('they'), score=0.0)])
    ut.assert_equal(list(ranked), words)

    scored, ranked = test(prevalence_weight=0, levenshtein_distance_weight=1)
    ut.assert_equal(list(scored),
                    [Score(Word('they'), score=1.0), Score(Word('them'), score=1.0), Score(Word('th'), score=1.0),
                     Score(Word('there'), score=0.8333333333333334), Score(Word('their'), score=0.8333333333333334),
                     Score(Word('though'), score=0.5), Score(Word('platypus'), score=0.0)])

    ut.assert_equal(list(ranked),
                    [Word('th'), Word('them'), Word('they'), Word('their'), Word('there'), Word('though'),
                     Word('platypus')])

if __name__ == '__main__' :
    from nlplib.general.unit_test import UnitTest
    __test__(UnitTest())

