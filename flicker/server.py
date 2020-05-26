from pygame import mixer  # Load the required library
from subprocess import call

import tornado.ioloop
import tornado.web
import alsaaudio
# curr_vol = 50


def change_vol(percent):
    call(["amixer", "-D", "pulse", "sset", "Master", "{}%".format(percent)])


def play(path):
    mixer.init()
    mixer.music.load(path)
    mixer.music.play()
    import time
    time.sleep(2)


def increase_vol():
    mixer = alsaaudio.Mixer()
    curr_vol = mixer.getvolume()[0]
    print(curr_vol)
    print(type(curr_vol))
    new_vol = curr_vol + 40
    if new_vol > 100:
        new_vol = 100

    change_vol(new_vol)
    curr_vol = new_vol


class WakeHandler(tornado.web.RequestHandler):
    def get(self):
        print("wake up man")
        play("/home/yonatan/Desktop/wakeup.mp3")


class LookHandler(tornado.web.RequestHandler):
    def get(self):
        print("look away from the screen")
        play("/home/yonatan/Desktop/lookaway.mp3")


def make_app():
    return tornado.web.Application([
        (r"/wake", WakeHandler),
        (r"/look", LookHandler)
    ])


if __name__ == "__main__":
    app = make_app()
    app.listen(8888)
    tornado.ioloop.IOLoop.current().start()

print(1)
