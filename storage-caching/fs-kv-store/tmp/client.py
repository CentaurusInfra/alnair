import pyinotify

class Client(pyinotify.ProcessEvent):
    def process_IN_CREATE(self, event):
        print('create {}'.format(event.pathname))

    def process_IN_MODIFY(self, event):
        print('modify {}'.format(event.pathname))

    def process_IN_READ(self, event):
        print('close nowrite {}'.format(event.pathname))


if __name__=="__main__":
    client = Client()
    wm = pyinotify.WatchManager()
    mask = pyinotify.IN_CREATE | pyinotify.IN_MODIFY | pyinotify.IN_CLOSE_NOWRITE
    wm.add_watch("/share/hello", mask)
    notifier = pyinotify.Notifier(wm, client)
    try:
        notifier.loop()
    except KeyboardInterrupt:
        notifier.stop()