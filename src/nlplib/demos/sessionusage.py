

if __name__ == '__main__' :

    import nlplib

    db = nlplib.Database() # By default this uses an SQLite in-memory database.

    with db as session :
        session.add(nlplib.Word('foo'))
        session.add(nlplib.Word('bar'))

    with db as session :
        print(list(session.access.vocabulary())) # Will print foo and bar.

    @db
    def print_vocabulary (session) :
        print(list(session.access.vocabulary())) # Will print foo and bar again.
    print_vocabulary()

    try :
        with db as session :
            session.add(nlplib.Word('baz'))
            session.add(nlplib.Word('qux'))
            raise IOError # An arbitrary exception, new additions are rolled back.
    except IOError :
        pass

    with db as session :
        print(list(session.access.vocabulary())) # Will print foo and bar, but not baz or qux.

