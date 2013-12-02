

__all__ = ['clean']

def clean (raw_string) :
    clean_string = ' '.join(str(raw_string).split()).strip().lower()
    return clean_string

