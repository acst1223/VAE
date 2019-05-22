from utils.log_file import log_file


@log_file('test.txt')
def f():
    print(123)
    print(123)
    print('abc')


f()
