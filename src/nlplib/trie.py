from nlplib.core.model import Database, Word
from nlplib.core.model.backend.sqlalchemy.map.natural_language import TrieNode

db = Database()

with db as session :
    for char in 'abcdefg' :
        session.add(Word(char))

with db as session :
    a, b, c, d, e, f, g = session.access.all_words()

    an = session.add(TrieNode(a))

    bn = an.add_child(TrieNode(b))
    cn = an.add_child(TrieNode(c))

    dn = bn.add_child(TrieNode(d))

def gram (session, words) :
    return session._sqlalchemy_session.query(TrieNode).\
        filter(TrieNode.seq_id==words[2].id).\
        join(TrieNode.parent, aliased=True).\
        filter(TrieNode.seq_id==words[1].id).\
        join(TrieNode.parent, aliased=True, from_joinpoint=True).\
        filter(TrieNode.seq_id==words[0].id).\
        all()

with db as session :

    last = session._sqlalchemy_session.query(TrieNode).filter(TrieNode.seq_id==session.access.word('a').id).one().children[0].children[0]

    print(session.access.specific(Word, last.id))


    print(session.access.specific(Word, gram(session, session.access.words('a b d'))[0].seq_id))