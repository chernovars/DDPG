from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from random import randrange
import time

class Plot:
    def __init__(self):
        self.x_data, self.y_data = [], []
        self.figure = plt.figure()
        self.line = plt.plot_date(self.x_data, self.y_data, '-')
        self.animation = None

    def __update(self, frame):
        self.line.set_data(self.x_data, self.y_data)
        self.figure.gca().relim()
        self.figure.gca().autoscale_view()

    def update(self, x, y, set_new=False):
        if self.animation == None:
            self.animation = FuncAnimation(self.figure, self.__update, interval=200)
            plt.show()
        if not set_new:
            self.x_data.append(x)
            self.y_data.append(y)
        else:
            self.x_data = x
            self.y_data = y

