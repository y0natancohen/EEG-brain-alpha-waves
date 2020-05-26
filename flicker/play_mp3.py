from pygame import mixer  # Load the required library
mixer.init()
mixer.music.load('/home/yonatan/Desktop/horn.mp3')
mixer.music.play()
import time
time.sleep(5)
