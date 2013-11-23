

__all__ = ['process']

def process (raw_string) :
    clean_string = ' '.join(str(raw_string).split()).strip().lower()
    return clean_string

