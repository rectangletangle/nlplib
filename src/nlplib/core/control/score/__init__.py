

from functools import total_ordering

from nlplib.general.math import normalize_values
from nlplib.general import pretty_float
from nlplib.core.base import Base

__all__ = ['WeightedFunction', 'Score', 'Scored', 'ScoredAgainst', 'weighted']

class WeightedFunction (Base) :
    ''' Weighted functions are functions that have an associated weight attribute. This allows for weighted scoring
        functions. '''

    __slots__ = ('function', 'weight', 'low_is_better')

    def __init__ (self, function, weight=1.0, low_is_better=False) :
        self.function = function

        self.weight = float(weight)
        self.low_is_better = low_is_better

    def __repr__ (self, *args, **kw) :
        return super().__repr__(self.function.__name__, str(hex(id(self))), *args, **kw)

    def __call__ (self, *args, **kw) :
        return self.function(*args, **kw)

    def adjust_orientation (self, normalized_score) :
        ''' This adjusts the orientation of the score (if necessary), seeing as a low score in some metrics indicates a
            "better match." '''

        return (1.0 - normalized_score) if self.low_is_better else normalized_score

@total_ordering
class Score (Base) :
    ''' This is an object designed to hold the scores and subscores for a particular object. '''

    __slots__ = ('object', 'score', 'subscores')

    def __init__ (self, object, score=0.0, subscores=None) :
        self.object = object

        self.score = score
        self.subscores = {} if subscores is None else subscores

    def __repr__ (self, *args, **kw) :
        return super().__repr__(pretty_float(self.score), self.object, *args, **kw)

    def __float__ (self) :
        return float(self.score)

    def __int__ (self) :
        return int(self.score)

    def _unorderable_error (self) :
        return TypeError('Value comparisons with a <Score> object must be against an object which can be converted to '
                         'a floating point number.')

    def __iter__ (self) :
        ''' This allows for use of Python's multiple assignment syntax. '''

        yield self.object
        yield self.score

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

class Scored (Base) :
    def __init__ (self, unnormalized_scores) :
        self.unnormalized_scores = list(unnormalized_scores)

    def __iter__ (self) :
        normalized_score_values = normalize_values(score.score for score in self.unnormalized_scores)

        for unnormalized_score, normalized_score_value in zip(self.unnormalized_scores, normalized_score_values) :
            normalized_score = Score(object=unnormalized_score.object, score=normalized_score_value)

            yield normalized_score

    def sorted (self) :
        # todo : This probably could be done in a more efficient manner.
        sorted_by_objects_only = sorted(self, key=lambda score : score.object)
        return sorted(sorted_by_objects_only, reverse=True)

    def ranked (self) :
        ''' This returns objects ordered by how high their score was. '''

        for scored in self.sorted() :
            yield scored.object

class ScoredAgainst (Scored) :
    ''' This class is used to generate scores normalized between 0 and 1 (1 indicating a better match), based on a
        weighted average of multiple sub-scores. '''

    def __init__ (self, object, similar_objects, weighted_scoring_functions=()) :
        self.object = object
        self.similar_objects = similar_objects
        self.weighted_scoring_functions = weighted_scoring_functions

    def __iter__ (self) :
        scores = [Score(similar_object) for similar_object in self.similar_objects]
        self._calculate_subscores(scores)
        return self._calculate_scores(scores)

    def _calculate_subscores (self, scores) :
        for scoring_function in self.weighted_scoring_functions :
            unnormalized_subscores = (scoring_function(self.object, score.object)
                                      for score in scores)

            normalized_subscores = normalize_values(unnormalized_subscores)

            for score, normalized_subscore in zip(scores, normalized_subscores) :
                score.subscores[scoring_function] = scoring_function.adjust_orientation(normalized_subscore)

    def _calculate_scores (self, scores) :
        total_of_function_weights = sum(scoring_function.weight
                                        for scoring_function in self.weighted_scoring_functions)

        for score in scores :
            total_of_subscores_for_object = sum(scoring_function.weight * subscore
                                                for scoring_function, subscore in score.subscores.items())

            try :
                score.score = total_of_subscores_for_object / total_of_function_weights
            except ZeroDivisionError :
                # There were probably no scoring functions.
                score.score = 1.0

            yield score

def weighted (function, *args, **kw) :
    return WeightedFunction(function, *args, **kw)

def __test__ (ut) :
    from nlplib.core.control.score.metric import levenshtein_distance, count
    from nlplib.core.model import Word

    word = Word('the')

    words = [Word('their', count=100),
             Word('platypus', count=60),
             Word('there', count=10),
             Word('them', count=4),
             Word('th', count=2),
             Word('they', count=1),
             Word('though', count=1)]

    def test (count_weight, levenshtein_distance_weight) :

        weighted_count = weighted(lambda object, similar : count(similar),
                                  weight=count_weight)

        weighted_levenshtein_distance = weighted(levenshtein_distance,
                                                 weight=levenshtein_distance_weight,
                                                 low_is_better=True)

        args = (word, words, [weighted_levenshtein_distance, weighted_count])

        scored = ScoredAgainst(*args)
        return (scored.sorted(), scored.ranked())

    # No scoring functions, means everything is a perfect match.
    ut.assert_equal(ScoredAgainst(word, words, []).sorted(),
                    [Score(Word('though'), score=1.0), Score(Word('they'), score=1.0), Score(Word('there'), score=1.0),
                     Score(Word('them'), score=1.0), Score(Word('their'), score=1.0), Score(Word('th'), score=1.0),
                     Score(Word('platypus'), score=1.0)])

    # Tests behavior when things are missing.
    ut.assert_equal(list(ScoredAgainst(word, [], [])), [])
    ut.assert_equal(list(ScoredAgainst(word, [],
                    [weighted(levenshtein_distance, weight=2.0, low_is_better=True)])), [])

    # Without scoring functions, this essentially just sorts the word objects alphabetically.
    ut.assert_equal(list(ScoredAgainst(word, words, []).ranked()), sorted(words))

    sorted_, ranked = test(count_weight=1, levenshtein_distance_weight=0)
    ut.assert_equal(sorted_,
                    [Score(Word('their'), score=1.0), Score(Word('platypus'), score=0.5959595959595959),
                     Score(Word('there'), score=0.09090909090909091), Score(Word('them'), score=0.030303030303030304),
                     Score(Word('th'), score=0.010101010101010102), Score(Word('though'), score=0.0),
                     Score(Word('they'), score=0.0)])
    ut.assert_equal(list(ranked), words)

    sorted_, ranked = test(count_weight=0, levenshtein_distance_weight=1)
    ut.assert_equal(sorted_,
                    [Score(Word('they'), score=1.0), Score(Word('them'), score=1.0), Score(Word('th'), score=1.0),
                     Score(Word('there'), score=0.8333333333333334), Score(Word('their'), score=0.8333333333333334),
                     Score(Word('though'), score=0.5), Score(Word('platypus'), score=0.0)])
    ut.assert_equal(list(ranked),
                    [Word('th'), Word('them'), Word('they'), Word('their'), Word('there'), Word('though'),
                     Word('platypus')])

if __name__ == '__main__' :
    from nlplib.general.unittest import UnitTest
    __test__(UnitTest(True))

