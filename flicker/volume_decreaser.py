import alsaaudio
from subprocess import call
import time

# curr_vol = 100

def change_vol(percent):
    call(["amixer", "-D", "pulse", "sset", "Master", "{}%".format(percent)])


def decrease_vol():
    mixer = alsaaudio.Mixer()
    curr_vol = mixer.getvolume()[0]
    new_vol = curr_vol - 4
    if new_vol < 0:
        new_vol = 0

    change_vol(new_vol)
    curr_vol = new_vol


while True:
    decrease_vol()
    time.sleep(1)

