from tfsnippet.utils import EventSource


def event_handler(**kwargs):
    print('event triggered: kwargs {}'.format(kwargs))


def event_handler_another(**kwargs):
    print('ringo uri: kwargs {}'.format(kwargs))


# use alone
class SomeObject(EventSource):

    def func(self, **kwargs):
        self.reverse_fire('some_event', **kwargs)


obj = SomeObject()
obj.on('some_event', event_handler)
obj.on('some_event', event_handler_another)
obj.func(apple='shoujo')
