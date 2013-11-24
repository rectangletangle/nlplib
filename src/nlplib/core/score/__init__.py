
from functools import total_ordering

from nlplib.general.math import normalize_values
from nlplib.core import Base

__all__ = ['WeightedFunction', 'Score', 'weighted', 'score', 'rank']

class WeightedFunction (Base) :
    ''' Weighted functions are functions that have an associated weight attribute. This allows for weighted scoring
        functions. '''

    __slots__ = ('function', 'weight', 'low_is_better')

    def __init__ (self, function, weight=1.0, low_is_better=False) :
        self.function      = function
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

    def __init__ (self, object) :
        self.object    = object
        self.score     = 0.0
        self.subscores = {}

    def __repr__ (self) :
        return super().__repr__(self.object, score=self.score)

    def __float__ (self) :
        return float(self.score)

    def __int__ (self) :
        return int(self.score)

    def __lt__ (self, other) :
        return self.score < float(other)

    def __eq__ (self, other) :
        return self.score == float(other)

def _calculate_subscores (weighted_functions, object, scores) :
    for weighted_function in weighted_functions :

        unnormalized_subscores = (weighted_function(object, score.object)
                                  for score in scores)

        normalized_subscores = normalize_values(unnormalized_subscores)

        for score, normalized_subscore in zip(scores, normalized_subscores) :
            score.subscores[weighted_function] = weighted_function.adjust_orientation(normalized_subscore)

def _calculate_scores (weighted_functions, scores) :

    total_of_function_weights = sum(weighted_function.weight
                                    for weighted_function in weighted_functions)

    for score in scores :
        total_of_subscores_for_object = sum(weighted_function.weight * subscore
                                            for weighted_function, subscore in score.subscores.items())

        try :
            score.score = total_of_subscores_for_object / total_of_function_weights
        except ZeroDivisionError :
            # There were probably no scoring functions.
            score.score = 1.0

def score (weighted_functions, object, similar_objects,
           sort=lambda scores : sorted(scores, key=lambda score : (float(score), str(score.object)), reverse=True)) :
    # todo : document

    scores = [Score(similar_object) for similar_object in similar_objects]

    try :
        _calculate_subscores(weighted_functions, object, scores)
    except ValueError :
        # This happens if the <scores> list is empty.
        return scores
    else :
        _calculate_scores(weighted_functions, scores)
        return sort(scores)

def rank (*args, **kw) :
    return (scored.object for scored in score(*args, **kw))

