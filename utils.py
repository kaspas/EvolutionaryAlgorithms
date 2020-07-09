#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 19:53:40 2019

@author: panagiotis
"""
import sys
sys.path.insert(1,'/home/panagiotis/Desktop/EvolutionaryAlgorithms/data/');
from WoodProblemDefinition import Stock, Order1, Order2, Order3
import math
from random import randint, random, seed
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as pltcol
from shapely.ops import cascaded_union
import shapely
from descartes import PolygonPatch

from matplotlib.gridspec import GridSpec
# Simple helper class for getting matplotlib patches from shapely polygons with different face colors
class PlotPatchHelper:
    # a colormap with 41 colors
    CMapColors = np.array([
            [0,0.447,0.741,1],
            [0.85,0.325,0.098,1],
            [0.929,0.694,0.125,1],
            [0.494,0.184,0.556,1],
            [0.466,0.674,0.188,1],
            [0.301,0.745,0.933,1],
            [0.635,0.078,0.184,1],
            [0.333333333,0.333333333,0,1],
            [0.333333333,0.666666667,0,1],
            [0.666666667,0.333333333,0,1],
            [0.666666667,0.666666667,0,1],
            [1,0.333333333,0,1],
            [1,0.666666667,0,1],
            [0,0.333333333,0.5,1],
            [0,0.666666667,0.5,1],
            [0,1,0.5,1],
            [0.333333333,0,0.5,1],
            [0.333333333,0.333333333,0.5,1],
            [0.333333333,0.666666667,0.5,1],
            [0.333333333,1,0.5,1],
            [0.666666667,0,0.5,1],
            [0.666666667,0.333333333,0.5,1],
            [0.666666667,0.666666667,0.5,1],
            [1,0,0.5,1],
            [1,0.333333333,0.5,1],
            [1,0.666666667,0.5,1],
            [1,1,0.5,1],
            [0,0.333333333,1,1],
            [0,0.666666667,1,1],
            [0,1,1,1],
            [0.333333333,0,1,1],
            [0.333333333,0.333333333,1,1],
            [0.333333333,0.666666667,1,1],
            [0.333333333,1,1,1],
            [0.666666667,0,1,1],
            [0.666666667,0.333333333,1,1],
            [0.666666667,0.666666667,1,1],
            [0.666666667,1,1,1],
            [1,0,1,1],
            [1,0.333333333,1,1],
            [1,0.666666667,1,1]
            ])
    
    
    # Alpha controls the opaqueness, Gamma how darker the edge line will be and LineWidth its weight
    def __init__(self, Gamma=1.3, Alpha=0.9, LineWidth=2.0):
        self.Counter = 0
        self.Gamma = Gamma          # darker edge color if Gamma>1 -> faceColor ** Gamma; use np.inf for black
        self.Alpha = Alpha          # opaqueness level (1-transparency)
        self.LineWidth = LineWidth  # edge weight
    
    # circles through the colormap and returns the FaceColor and the EdgeColor (as FaceColor^Gamma)
    def nextcolor(self):
        col = self.CMapColors[self.Counter,:].copy()
        self.Counter = (self.Counter+1) % self.CMapColors.shape[0]
        return (col, col**self.Gamma)
    
    # returns a list of matplotlib.patches.PathPatch from the provided shapely polygons, using descartes; a list is 
    # returned even for a single polygon for common handling
    def get_patches(self, poly):
        if not isinstance(poly, list): # single polygon, make it a one element list for common handling
            poly = [poly]
        patchList = []
        for p in poly:
            fCol, eCol = self.nextcolor()
            patchList.append(PolygonPatch(p, alpha=self.Alpha, FaceColor=fCol, EdgeColor=eCol, 
                                          LineWidth=self.LineWidth))        
        return patchList


# Plots one or more shapely polygons in the provided axes ax. The named parameter values **kwargs are passed into
# PlotPatchHelper's constructor, e.g. you can write plotShapelyPoly(ax, poly, LineWidth=3, Alpha=1.0). Returns a list
# with the drawn patches objects even for a single polygon, for common handling
def plotShapelyPoly(ax, poly, **kwargs):
    return [ax.add_patch(p) for p in PlotPatchHelper(**kwargs).get_patches(poly)]

#class FigureObjects:
#    """ Class for storing and updating the figure's objects.
#        
#        The initializer creates the figure given only the lower and upper bounds (scalars, since the bounds are 
#        typically equal in both dimensions).
#        
#        The update member function accepts a DEGL object and updates all elements in the figure.
#        
#        The figure has a top row of 1 subplots. This shows the best-so-far global finess value .
#        The bottom row shows the global best-so-far solution achieved by the algorithm and the remaining current stock after placement.
#    """
#    
#    def __init__(self, LowerBound, UpperBound,stock):
#        """ Creates the figure that will be updated by the update member function.
#            
#        All line objects (best solution,, global fitness line,etc) are initialized with NaN values, as we only 
#        setup the style. Best-so-far fitness 
#        
#        The input arguments LowerBound & UpperBound must be scalars, otherwise an assertion will fail.
#        """
#        self.plots =[]
#        if isinstance(stock,list):
#            self.Total=2*len(stock)
#            self.Cols = 4
#            self.Rows = (self.Total-1) // self.Cols 
#            self.Rows += (self.Total-1) % self.Cols+1
#            self.Position = range(1,len(stock) + 1)
#            self.grid=plt.GridSpec(self.Rows,self.Cols)
#
#            self.fig = plt.figure()
#            self.ax=[]
#
#
#            for i in range(len(stock)//2+1):
#                if i == 0:
#                    self.ax.append(plt.subplot(self.grid[0,0:],
#                        title=('Best-so-far global best fitness: {:g}'.format(np.nan))))
#                    self.lineBestFit, = self.ax[0].plot([], [])
#                else:
#                    for j in range(4):
#                        if j%2==0:
#                            self.ax.append(plt.subplot(self.grid[i,j],
#                            title='Rotated & translated order in stock '+str(i)))
#                        else:
#                            self.ax.append(plt.subplot(self.grid[i,j],
#                            title='Remaining after set difference in stock '+str(i)))
#    
#            '''
#            for i in range(len(stock)+1):
#                if i==0:
#                    self.ax.append(plt.subplot(self.grid[0,0:],
#                        title=('Best-so-far global best fitness: {:g}'.format(np.nan))))
#                    self.lineBestFit, = self.ax[0].plot([], [])
#
#                else:
#                    self.ax.append(plt.subplot(self.grid[i,0],
#                        title='Rotated & translated order in stock '+str(i)))
#                    self.ax.append(plt.subplot(self.grid[i,1],
#                        title='Remaining after set difference in stock '+str(i)))
#                    '''
#        
#        
#        else:
#            self.plots.append(211)
#            self.plots.append(223)
#            self.plots.append(224)
#            self.fig = plt.figure()
#            self.ax=[1,2,3]
#    
#            self.ax[0] = plt.subplot(211)
#    
#            self.ax[0].set_title('Best-so-far global best fitness: {:g}'.format(np.nan))
#            self.lineBestFit, = self.ax[0].plot([], [])
#            
#            # auto-arrange subplots to avoid overlappings and show the plot
#            # 3 subplots : 1: fitness , 2: newOrder, 3: Remaining (for current fitness and positions)
#            self.ax[1] = plt.subplot(223)
#            self.ax[1].set_title('Rotated & translated order')
#            self.ax[2] = plt.subplot(224)
#            self.ax[2].set_title('Remaining after set difference')
#            self.fig.tight_layout()
#            
#        
#        self.fig.tight_layout()
#
#    
#    def update(self, deglObject,stock,order,numberOfVar):
#        """ Updates the figure in each iteration provided a DEGL object. """
#        # pso.Iteration is the PSO initialization; setup the best-so-far fitness line xdata and ydata, now that 
#        # we know MaxIterations
#        
#        #Changes in plot in order to plot transformed order and remainings on every best fitness update
#        if deglObject.Iteration == -1:
#            xdata = np.arange(deglObject.MaxIterations+1)-1
#            self.lineBestFit.set_xdata(xdata)
#            self.lineBestFit.set_ydata(deglObject.GlobalBestSoFarFitnesses)
#       
#        # update the global best fitness line (remember, -1 is for initialization == iteration 0)
#        if isinstance(stock,list):
#            self.lineBestFit.set_ydata(deglObject.GlobalBestSoFarFitnesses)
#            self.ax[0].relim()
#            self.ax[0].autoscale_view()
#            self.ax[0].title.set_text('Best-so-far global best fitness: {:g}'.format(deglObject.GlobalBestFitness))
#
#            for i,s in enumerate(stock):
#                
#                self.ax[(i+1)*2-1].cla()
#                self.ax[(i+1)*2].cla()
#                self.ax[(i+1)*2-1].set_title('Rotated & translated order in stock '+str((i+1)))
#                self.ax[(i+1)*2].set_title('Remaining after set difference in stock '+str((i+1)))
#                p=deglObject.GlobalBestPosition
#                newOrder=[]
#
#                for j in range(len(order)):
#                    
#                    if int(p[j*numberOfVar+3])==i:
#                        newOrder.append(shapely.affinity.rotate(
#                        shapely.affinity.translate(
#                                order[j],xoff=p[j*numberOfVar+0],yoff=p[j*numberOfVar+1]),
#                                p[j*numberOfVar+2],origin = 'centroid'))
#                new = shapely.ops.cascaded_union(newOrder)    
#                remaining = s.difference(new)
#                pp = plotShapelyPoly(self.ax[(i+1)*2-1], [s] + newOrder)
#                pp[0].set_facecolor([1,1,1,1])
#                plotShapelyPoly(self.ax[(i+1)*2], remaining)
#                
#
#                self.ax[(i+1)*2-1].relim()
#                self.ax[(i+1)*2-1].autoscale_view()
#                self.ax[(i+1)*2].set_xlim(self.ax[(i+1)*2-1].get_xlim())
#                self.ax[(i+1)*2].set_ylim(self.ax[(i+1)*2-1].get_ylim())
#            
#        else:
#            self.lineBestFit.set_ydata(deglObject.GlobalBestSoFarFitnesses)
#            self.ax[0].relim()
#            self.ax[0].autoscale_view()
#            self.ax[0].title.set_text('Best-so-far global best fitness: {:g}'.format(deglObject.GlobalBestFitness))
#            p=deglObject.GlobalBestPosition
#            newOrder= [shapely.affinity.rotate(
#                shapely.affinity.translate(
#                        order[i],xoff=p[i*4+0],yoff=p[i*4+1]),p[i*4+2],origin = 'centroid') for i in range(len(order))]
#            new = shapely.ops.cascaded_union(newOrder)    
#            remaining = stock.difference(new)
#            
#            
#            self.ax[1].cla()
#            self.ax[2].cla()
#            self.ax[1].set_title('Rotated & translated order')
#            self.ax[2].set_title('Remaining after set difference')
#            #stock=stock.buffer(-0.7,join_style = shapely.geometry.JOIN_STYLE.mitre)
#            #stock=stock.buffer(0.7,join_style = shapely.geometry.JOIN_STYLE.mitre)
#            pp = plotShapelyPoly(self.ax[1], [stock] + newOrder)
#            pp[0].set_facecolor([1,1,1,1])
#            if(remaining.is_empty==False):    
#                plotShapelyPoly(self.ax[2], remaining)
#            self.ax[1].relim()
#            self.ax[1].autoscale_view()
#            self.ax[2].set_xlim(self.ax[1].get_xlim())
#            self.ax[2].set_ylim(self.ax[1].get_ylim())
#        
#        self.fig.tight_layout()
#        self.fig.canvas.draw()
#        self.fig.canvas.flush_events()
class FigureObjects:
    """ Class for storing and updating the figure's objects.
        
        The initializer creates the figure given only the lower and upper bounds (scalars, since the bounds are 
        typically equal in both dimensions).
        
        The update member function accepts a DEGL object and updates all elements in the figure.
        
        The figure has a top row of 1 subplots. This shows the best-so-far global finess value .
        The bottom row shows the global best-so-far solution achieved by the algorithm and the remaining current stock after placement.
    """
    
    def __init__(self,stock):
        """ Creates the figure that will be updated by the update member function.
            
        All line objects (best solution,, global fitness line,etc) are initialized with NaN values, as we only 
        setup the style. Best-so-far fitness 
        
        The input arguments LowerBound & UpperBound must be scalars, otherwise an assertion will fail.
        """
        self.plots =[]
        if isinstance(stock,list):
            if len(stock)>1:
                
                self.Total=len(stock)
                self.Cols = 4
                self.Rows = self.Total//self.Cols+1
                if self.Rows ==1:
                    self.Rows=2
                if self.Total<self.Cols:
                    self.Cols=self.Total
                self.heights = [1]#, 3, 3]
                for i in range(self.Rows-1):
                    self.heights.append(3)
                self.Position = range(1,len(stock) + 1)
                self.grid=plt.GridSpec(self.Rows,self.Cols,height_ratios=self.heights)
    
                self.fig = plt.figure()
                self.ax=[]
    
    
                for i in range(self.Rows):
                    if i == 0:
                        self.ax.append(plt.subplot(self.grid[0,0:],
                            title=('Best-so-far global best fitness: {:g}'.format(np.nan))))
                        self.lineBestFit, = self.ax[0].plot([], [])
                    else:
                        for j in range(self.Cols):
                            self.ax.append(plt.subplot(self.grid[i,j],
                            title='Rotated & translated order in stock '+str(i+j)))
            else:
                self.plots.append(211)
                self.plots.append(223)
                self.plots.append(224)
                self.fig = plt.figure()
                self.ax=[1,2,3]
        
                self.ax[0] = plt.subplot(211)
        
                self.ax[0].set_title('Best-so-far global best fitness: {:g}'.format(np.nan))
                self.lineBestFit, = self.ax[0].plot([], [])
                
                # auto-arrange subplots to avoid overlappings and show the plot
                # 3 subplots : 1: fitness , 2: newOrder, 3: Remaining (for current fitness and positions)
                self.ax[1] = plt.subplot(223)
                self.ax[1].set_title('Rotated & translated order')
                self.ax[2] = plt.subplot(224)
                self.ax[2].set_title('Remaining after set difference')
                self.fig.tight_layout()
        
        else:
            self.plots.append(211)
            self.plots.append(223)
            self.plots.append(224)
            self.fig = plt.figure()
            self.ax=[1,2,3]
    
            self.ax[0] = plt.subplot(211)
    
            self.ax[0].set_title('Best-so-far global best fitness: {:g}'.format(np.nan))
            self.lineBestFit, = self.ax[0].plot([], [])
            
            # auto-arrange subplots to avoid overlappings and show the plot
            # 3 subplots : 1: fitness , 2: newOrder, 3: Remaining (for current fitness and positions)
            self.ax[1] = plt.subplot(223)
            self.ax[1].set_title('Rotated & translated order')
            self.ax[2] = plt.subplot(224)
            self.ax[2].set_title('Remaining after set difference')
            self.fig.tight_layout()
            
        
        self.fig.tight_layout()

    
    def update(self, deglObject,stock,indx,order,numberOfVar):
        """ Updates the figure in each iteration provided a DEGL object. """
        # pso.Iteration is the PSO initialization; setup the best-so-far fitness line xdata and ydata, now that 
        # we know MaxIterations
        
        #Changes in plot in order to plot transformed order and remainings on every best fitness update
        if deglObject.Iteration == -1:
            xdata = np.arange(deglObject.MaxIterations+1)-1
            self.lineBestFit.set_xdata(xdata)
            self.lineBestFit.set_ydata(deglObject.GlobalBestSoFarFitnesses)
       
        
        # update the global best fitness line (remember, -1 is for initialization == iteration 0)
        if isinstance(stock,list):
            if len(stock)>1:
                    
                self.lineBestFit.set_ydata(deglObject.GlobalBestSoFarFitnesses)
                self.ax[0].relim()
                self.ax[0].autoscale_view()
                self.ax[0].title.set_text('Best-so-far global best fitness: {:g}'.format(deglObject.GlobalBestFitness))
    
                for i,s in enumerate(stock):
                    
                    self.ax[(i+1)].cla()
                    
                    self.ax[(i+1)].set_title('Rotated & translated order in stock '+str((i+1)))
                    p=deglObject.GlobalBestPosition
                    newOrder=[]
                    if indx ==i:
                        for j in range(len(order)):
    #                if int(p[j*numberOfVar+3])==i:
                            newOrder.append(shapely.affinity.rotate(
                                    shapely.affinity.translate(
                                        order[j],xoff=p[j*numberOfVar+0],yoff=p[j*numberOfVar+1]),
                                        p[j*numberOfVar+2],origin = 'centroid'))
                    
                        pp = plotShapelyPoly(self.ax[(i+1)], [s] + newOrder)
                    else:
                        pp = plotShapelyPoly(self.ax[(i+1)], [s] )
                    pp[0].set_facecolor([1,1,1,1])
                    
                    self.ax[(i+1)].relim()
                    self.ax[(i+1)].autoscale_view()
            else:
                self.lineBestFit.set_ydata(deglObject.GlobalBestSoFarFitnesses)
                self.ax[0].relim()
                self.ax[0].autoscale_view()
                self.ax[0].title.set_text('Best-so-far global best fitness: {:g}'.format(deglObject.GlobalBestFitness))
                p=deglObject.GlobalBestPosition
                newOrder= [shapely.affinity.rotate(
                    shapely.affinity.translate(
                            order[i],xoff=p[i*numberOfVar+0],yoff=p[i*numberOfVar+1]),
                                p[i*numberOfVar+2],origin = 'centroid') for i in range(len(order))]
                new = shapely.ops.cascaded_union(newOrder)    
                remaining = stock[0].difference(new)
                
                
                self.ax[1].cla()
                self.ax[2].cla()
                self.ax[1].set_title('Rotated & translated order')
                self.ax[2].set_title('Remaining after set difference')
                #stock=stock.buffer(-0.7,join_style = shapely.geometry.JOIN_STYLE.mitre)
                #stock=stock.buffer(0.7,join_style = shapely.geometry.JOIN_STYLE.mitre)
                pp = plotShapelyPoly(self.ax[1], stock + newOrder)
                pp[0].set_facecolor([1,1,1,1])
                if(remaining.is_empty==False):    
                    plotShapelyPoly(self.ax[2], remaining)
                self.ax[1].relim()
                self.ax[1].autoscale_view()
                self.ax[2].set_xlim(self.ax[1].get_xlim())
                self.ax[2].set_ylim(self.ax[1].get_ylim())
            
        
                    
        else:
            self.lineBestFit.set_ydata(deglObject.GlobalBestSoFarFitnesses)
            self.ax[0].relim()
            self.ax[0].autoscale_view()
            self.ax[0].title.set_text('Best-so-far global best fitness: {:g}'.format(deglObject.GlobalBestFitness))
            p=deglObject.GlobalBestPosition
            newOrder= [shapely.affinity.rotate(
                shapely.affinity.translate(
                        order[i],xoff=p[i*4+0],yoff=p[i*4+1]),p[i*4+2],origin = 'centroid') for i in range(len(order))]
            new = shapely.ops.cascaded_union(newOrder)    
            remaining = stock.difference(new)
            
            
            self.ax[1].cla()
            self.ax[2].cla()
            self.ax[1].set_title('Rotated & translated order')
            self.ax[2].set_title('Remaining after set difference')
            #stock=stock.buffer(-0.7,join_style = shapely.geometry.JOIN_STYLE.mitre)
            #stock=stock.buffer(0.7,join_style = shapely.geometry.JOIN_STYLE.mitre)
            pp = plotShapelyPoly(self.ax[1], [stock] + newOrder)
            pp[0].set_facecolor([1,1,1,1])
            if(remaining.is_empty==False):    
                plotShapelyPoly(self.ax[2], remaining)
            self.ax[1].relim()
            self.ax[1].autoscale_view()
            self.ax[2].set_xlim(self.ax[1].get_xlim())
            self.ax[2].set_ylim(self.ax[1].get_ylim())
        
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        
def showResult(p,stock,order,numberOfVar,ind=0):
        """ Updates the figure in each iteration provided a DEGL object. """
        # pso.Iteration is the PSO initialization; setup the best-so-far fitness line xdata and ydata, now that 
        # we know MaxIterations
        
        #Changes in plot in order to plot transformed order and remainings on every best fitness update
        
       
        # update the global best fitness line (remember, -1 is for initialization == iteration 0)
        
        plots =[]
        if isinstance(stock,list):
            
            Total=len(stock)
            Cols = 4
            Rows = Total//Cols
            if Total<Cols:
                Cols=Total
            if Rows==0:
                Rows=1
            Position = range(1,len(stock))
            grid=plt.GridSpec(Rows,Cols)

            fig = plt.figure()
            ax=[]


            for i in range(Rows):
                    for j in range(Cols):
                        ax.append(plt.subplot(grid[i,j],
                        title='Rotated & translated order in stock '+str(ind+i+j+1)))
                    
        
        else:
            plots.append(121)
            plots.append(122)

            fig = plt.figure()
            ax=[1,2]
    

            # auto-arrange subplots to avoid overlappings and show the plot
            # 3 subplots : 1: fitness , 2: newOrder, 3: Remaining (for current fitness and positions)
            ax[0] = plt.subplot(121)
            ax[0].set_title('Rotated & translated order')
            ax[1] = plt.subplot(122)
            ax[1].set_title('Remaining after set difference')
            
        
        fig.tight_layout()
        
        
        
        
        if isinstance(stock,list):

            for i,s in enumerate(stock):
                
                ax[(i)].cla()
                
                ax[(i)].set_title('Rotated & translated order in stock '+str((ind+i+1)))
                newOrder=[]
                for j in range(len(order)):
                    if int(p[j*numberOfVar+3])==i:
                        newOrder.append(shapely.affinity.rotate(
                        shapely.affinity.translate(
                                order[j],xoff=p[j*numberOfVar+0],yoff=p[j*numberOfVar+1]),
                                p[j*numberOfVar+2],origin = 'centroid'))
                
                pp = plotShapelyPoly(ax[(i)], [s] + newOrder)
                pp[0].set_facecolor([1,1,1,1])
                
                ax[(i)].relim()
                ax[(i)].autoscale_view()
                
            
        else:
            
            newOrder= [shapely.affinity.rotate(
                shapely.affinity.translate(
                        order[i],xoff=p[i*4+0],yoff=p[i*4+1]),p[i*4+2],origin = 'centroid') for i in range(len(order))]
            new = shapely.ops.cascaded_union(newOrder)    
            remaining = stock.difference(new)
            
            
            ax[0].cla()
            ax[1].cla()
            ax[0].set_title('Rotated & translated order')
            ax[1].set_title('Remaining after set difference')
            #stock=stock.buffer(-0.7,join_style = shapely.geometry.JOIN_STYLE.mitre)
            #stock=stock.buffer(0.7,join_style = shapely.geometry.JOIN_STYLE.mitre)
            pp = plotShapelyPoly(ax[0], [stock] + newOrder)
            pp[0].set_facecolor([1,1,1,1])
            if(remaining.is_empty==False):    
                plotShapelyPoly(ax[1], remaining)
            ax[0].relim()
            ax[0].autoscale_view()
            ax[1].set_xlim(ax[0].get_xlim())
            ax[1].set_ylim(ax[0].get_ylim())
        
        fig.tight_layout()
        fig.canvas.draw()
        fig.canvas.flush_events()

def OutputFcn(de, figObj,stock,indx,order,numberOfVar):
    """ Our output function: updates the figure object and prints best fitness on terminal.
        
        Always returns False (== don't stop the iterative process)
    """
    if de.Iteration == -1:
        print('Iter.    Global best')
    print('{0:5d}    {1:.5f}'.format(de.Iteration, de.GlobalBestFitness))
    
    figObj.update(de,stock,indx,order,numberOfVar)
    
    return False


    
