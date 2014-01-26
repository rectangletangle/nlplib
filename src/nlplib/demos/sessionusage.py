''' A demonstration of how to use a session object to interact with a database. '''


import nlplib

db = nlplib.Database() # By default this uses an SQLite in-memory database.

# Using the database instance as a decorator supplies a session object to the function, when it's called.
@db
def print_vocabulary (session) :
    print(list(session.access.vocabulary()))

if __name__ == '__main__' :

    with db as session :
        session.add(nlplib.Word('foo'))
        session.add(nlplib.Word('bar'))

    with db as session :
        print(list(session.access.vocabulary())) # Will print foo and bar.

    print_vocabulary() # Will print foo and bar again.

    try :
        with db as session :
            session.add(nlplib.Word('baz'))
            session.add(nlplib.Word('qux'))
            raise ArithmeticError # An arbitrary exception, new additions are rolled back.
    except ArithmeticError :
        pass

    with db as session :
        print(list(session.access.vocabulary())) # Will print foo and bar, but not baz or qux.

