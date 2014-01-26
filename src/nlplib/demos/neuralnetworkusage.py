''' A demonstration of the neural network machine learning algorithm. '''


import random

import nlplib

# Neural network connection weights are initialized pseudorandomly, this call makes the results deterministic.
# This isn't necessary but, can be useful for testing.
random.seed(0)

if __name__ == '__main__' :

    nn = nlplib.NeuralNetwork(['a', 'b', 'c'], 4, ['d', 'e', 'f'],
                              name='some neural network')

    # These scores are pretty worthless because the network hasn't been trained yet.
    print('before training')
    for score in nn.predict(('a', 'b')) :
        print(score)

    print()

    # Do some training!
    rate = 0.2
    for _ in range(100) :
        nn.train(('a', 'b'), ('f',), rate=rate)
        nn.train(('b'), ('e',), rate=rate)

    # "f" gets the highest score here, as expected.
    print('testing a and b')
    for score in nlplib.Scored(nn.predict(('a', 'b'))).sorted() :
        print(score)

    print()

    # "e" gets the highest score here.
    print('testing only b')
    for score in nlplib.Scored(nn.predict(('b',))).sorted() :
        print(score)

    print()

    # "f" is a reasonable guess, seeing as the network has never seen "a" on its own before.
    print('testing only a')
    for score in nlplib.Scored(nn.predict(('a',))).sorted() :
        print(score)

    print()

    # Storing the network in the database is pretty straight forward.
    db = nlplib.Database()
    with db as session :
        session.add(nn)

    with db as session :
        # Here we retrieve the network from the database. The shortened form <session.access.nn> can also be used here.
        nn_from_db = session.access.neural_network('some neural network')

        print('testing a and b, again')
        for score in nlplib.Scored(nn_from_db.predict(('a', 'b'))).sorted() :
            print(score)

