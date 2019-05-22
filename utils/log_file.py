import functools
import sys
import datetime


class FileAppender(object):
    def __init__(self, file):
        self.file = file

    def write(self, st):
        with open(self.file, 'a') as f:
            st = str(st)
            if st != '\n':
                f.write('[%s] %s' % (str(datetime.datetime.now()), st))
            else:
                f.write('\n')


def log_file(file):
    def log_file_decorator(f):
        @functools.wraps(f)
        def wrapped_function(*args, **kwargs):
            old_stdout = sys.stdout
            sys.stdout = FileAppender(file)
            f(*args, **kwargs)
            sys.stdout = old_stdout
        return wrapped_function
    return log_file_decorator
