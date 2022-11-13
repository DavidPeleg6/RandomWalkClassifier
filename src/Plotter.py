# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 18:48:26 2020

@author: gedadav
"""
import matplotlib.pyplot as plt
from datetime import datetime
import os



def save_plot(location, title, fig):
    now = datetime.now()
    current_time = now.strftime(" %H_%M")
    figure_name = f'{location}{title}/time{current_time}.jpg'
    try:
        os.mkdir(location + title)
    except FileExistsError:
        pass
    fig.savefig(figure_name, optimize=True, dpi=900)


class Plotter:
    def __init__(self, interval=1, title='no title', x='epochs', y='loss'):
        self.counter = 0
        self.interval = interval
        self.data = []
        self.title = title
        self.x = x
        self.y = y
        
    def update(self, data):
        self.counter += 1
        if self.counter % self.interval == 0:
            self.data.append(data)
            
    def plot(self, save_name=None):
        # plotting stuff:
        fig = plt.figure()
        # plt.title(f"{self.title}\noptimization steps:{len(self.data) * self.interval}")
        plt.plot([i for i in range(len(self.data))], self.data)
        plt.xlabel(self.x)
        plt.ylabel(self.y)
        # plt.savefig("rand_walk" + str(n) + ".png", bbox_inches="tight", dpi=600)
        if save_name:
            save_plot(save_name, fig)
        fig.show()
        
        now = datetime.now()
        current_time = now.strftime(" %H_%M")
        new_dir = save_name + current_time
        try:
            os.mkdir(new_dir)
        except FileExistsError:
            figure_name = f'{new_dir}/accuracy.jpg'
        else:
            figure_name = f'{new_dir}/loss.jpg'
        fig.savefig(figure_name, optimize=True, dpi=900)

      
class TwoValPlotter:
    def __init__(self, interval=1, title='no title', x='epochs', y='y_axis', label1='data1', label2='data2'):
        self.counter = 0
        self.interval = interval
        self.data1,  self.data2 = [], []
        self.title = title
        self.x = x
        self.y = y
        self.label1 = label1
        self.label2 = label2
               
    def update(self, data1, data2):
        self.counter += 1
        if self.counter % self.interval == 0:
            self.data1.append(data1)
            self.data2.append(data2)
            
    def plot(self, location=None):
        # plotting stuff:
        fig = plt.figure()
        plt.plot([i for i in range(len(self.data1))], self.data1, label=self.label1)
        plt.plot([i for i in range(len(self.data2))], self.data2, label=self.label2)
        plt.xlabel(self.x)
        plt.ylabel(self.y)
        # plt.title(f"{self.title}")
        plt.legend()
        # plt.savefig("rand_walk" + str(n) + ".png", bbox_inches="tight", dpi=600)
        if location:
            save_plot(location, self.title, fig)
        fig.show()
        
    def update_title(self, title):
        self.title = title
        
        
        
class HorizontalTwoValPlotter:
    def __init__(self, plot_amount, interval=1, subtit=None, x='epochs', y='y_axis', label1='data1', label2='data2'):
        self.counter = 0
        self.interval = interval
        self.data1 = [[] for i in range(plot_amount)]
        self.data2 = [[] for i in range(plot_amount)]
        self.x = x
        self.y = y
        self.label1 = label1
        self.label2 = label2
        self.plots = plot_amount
        self.subtit = ['no tit' for i in range(plot_amount)] if not subtit else subtit
    
    """
    @params:
        data1= the list of loss for all networks
        data2= the list of accuracies for all networks
    """         
    def update(self, data1, data2):
        self.counter += 1
        if self.counter % self.interval == 0:
            for i in range(self.plots):
                self.data1[i].append(data1[i])
                self.data2[i].append(data2[i])
            
    def plot(self, save_name=None):
        # plotting stuff:
        fig, axs = plt.subplots(nrows=1, ncols=self.plots, sharey=True)
        for j in range(self.plots):    
            axs[j].plot([i for i in range(len(self.data1[j]))], self.data1[j], label=self.label1)
            axs[j].plot([i for i in range(len(self.data1[j]))], self.data2[j], label=self.label2)
            axs[j].set(title=self.subtit[j])
            axs[j].legend()
        axs[0].set(xlabel=self.x, ylabel=self.y)
        if save_name:
            save_plot(save_name, fig)
        fig.show()
        
    