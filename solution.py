#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 22:28:34 2020

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
from DEGL import DEGL
from DynNeighborPSO import DynNeighborPSO as PSO
from WoodProblemDefinition import Stock, Order1, Order2, Order3 
from itertools import permutations 
from utils import FigureObjects 
from utils import OutputFcn,showResult
from objectiveFunctions import ObjectiveFcn2
import time
from scipy.optimize import minimize
from noisyopt import minimizeCompass


optima=''

def isValid(p,stock,order,nVars):
    
    o = [shapely.affinity.rotate(order[i],p[i*nVars+2],origin = 'centroid') for i in range(len(order))]
    o= [shapely.affinity.translate(o[i],xoff=p[i*nVars + 0],yoff=p[i*nVars+1]) for i in range(len(order))]
    
    cascade = shapely.ops.cascaded_union(o)
    outOfBounds = cascade.difference(stock)
    if outOfBounds.area>0.0001:
        return False,cascade
    for o1 in range(len(o)):
        for o2 in range(len(o)):
            if o[o1]!=o[o2]:
                if abs(o[o1].area-o[o1].difference(o[o2]).area)>0.0001:
                    return False,cascade
    return True,cascade


def submit(order,stock,i,figObj,n):
    nVars = n*len(order)
    LowerBounds = np.zeros(nVars)
    UpperBounds = np.zeros(nVars)
    boundsStock = stock[i].bounds
    for j in range(len(order)):
        LowerBounds[j*n] = boundsStock[0]
        LowerBounds[j*n+1] = boundsStock[1]
        LowerBounds[j*n+2] = 0
        UpperBounds[j*n] = boundsStock[2]
        UpperBounds[j*n+1] = boundsStock[3]
        UpperBounds[j*n+2] = 360

    objFcn = lambda x: ObjectiveFcn2(x,stock[i],order,1,15,1000000,1000000,2,n)
    # lambda functor (unnamed function) so that the output function appears to accept one argument only, the 
    # DynNeighborPSO object; behind the scenes, the local object figObj is stored within the lambda
    outFun = lambda x: OutputFcn(x, figObj,stock,i,order,n)
    if optima == 'PSO' or optima == 'DEGL':
        if optima=='PSO':
            optimizer = PSO(objFcn, nVars, LowerBounds=LowerBounds, UpperBounds=UpperBounds, 
                            OutputFcn=outFun, UseParallel=False, 
                            MaxStallIterations=15,FunctionTolerance=10**(-3))
        elif optima == 'DEGL':
            optimizer = DEGL(objFcn, nVars, LowerBounds=LowerBounds, UpperBounds=UpperBounds, 
                            OutputFcn=outFun, UseParallel=False, 
                            MaxStallIterations=15,FunctionTolerance=10**(-3))
        optimizer.optimize()
        return optimizer

    else:
        lbMatrix = LowerBounds
        ubMatrix = UpperBounds
        bRangeMatrix = ubMatrix - lbMatrix
        x0 = lbMatrix + np.random.rand(1,nVars) * bRangeMatrix 
        if optima == 'Nelder-Mead':
            optimizer = minimize(objFcn, x0,
                     method='Nelder-Mead', tol=1e-6,options={'maxiter': 10000,'disp': True})
            
        if optima == 'L-BFGS-B':
            optimizer = minimize(objFcn, x0,
                     method='L-BFGS-B', tol=1e-6,options={'maxiter': 10000,'disp': True})
        if optima == 'SLSQP':
            optimizer = minimize(objFcn, x0,
                     method='SLSQP', tol=1e-6,options={'maxiter': 10000,'disp': True})
        if optima == 'PatternSearch':
            optimizer = minimizeCompass(objFcn, x0=x0[0], deltatol=0.1, paired=False,errorcontrol=False)
        return optimizer


def subsolution(S,Order,figObj,n): 
    stock= S.copy()
    ind =0
    p=[]
    O=[]
    iterations=[]
    
    submit_list=[]
    red_flag=0
    order = sorted(Order,key=lambda i:i.area, reverse=False)
    while order:
        s = sorted(enumerate(stock),key= lambda i:i[1].area,reverse=False)
        order = sorted(order,key=lambda i:i.area, reverse=False)
        area=s[ind][1].area
        for item in order:
            if item.area<=0.9*area:
                area-=item.area
                submit_list.append(item)
        for item in submit_list:
            order.remove(item)
        while submit_list:
            
            if optima =='PSO' or optima == 'DEGL':
                opt = submit(submit_list,stock,s[ind][0],figObj,n)
                valid,union = isValid(opt.GlobalBestPosition,s[ind][1],submit_list,n)
                iterations.append(opt.Iteration)
                if valid == False:
                    order.append(submit_list.pop())  
                    
                else:
                    red_flag = 0
                    stock[s[ind][0]]=stock[s[ind][0]].difference(union)
                    count =0
                    for i in opt.GlobalBestPosition:
                        p.append(i)
                        count=count+1
                        if count ==3:
                            p.append(s[ind][0])
                            count=0
                    O=O+submit_list
                    submit_list=[] 
            else:
                opt = submit(submit_list,stock,s[ind][0],figObj,n)
                valid,union = isValid(opt.x,s[ind][1],submit_list,n)
                iterations.append(opt.nit)
                if valid == False:
                    order.append(submit_list.pop())  
                    
                else:
                    red_flag = 0
                    stock[s[ind][0]]=stock[s[ind][0]].difference(union)
                    count =0
                    for i in opt.x:
                        p.append(i)
                        count=count+1
                        if count ==3:
                            p.append(s[ind][0])
                            count=0
                    O=O+submit_list
                    submit_list=[] 
        ind=ind+1
        if ind>= len(stock):
            red_flag=red_flag+1
            ind=0
        if red_flag==2:
            print("Den xwresan ola")
            break;
    return p,O,stock,iterations


def solution(S,Orders):
    stock=S.copy()
    n=3
    Ord=[]
    r=[]
    plt.ion()
    iterations=[]
    figObj = FigureObjects(stock)
    for order in Orders:
        opt,o,stock,itera=subsolution(stock,order,figObj,n)
        Ord=Ord+o
        for i in opt:
            r.append(i)
        for i in itera:
            iterations.append(i)
    return r,Ord,iterations






def solution4(S,orders):
    stock=S.copy()
    p=[]
    o=[]
    n=3
    figObj = FigureObjects(stock)
    Orders=orders.copy()
    
    
    
    for order in Orders:
        submit_list = order.copy()
        order =[]
        
        while submit_list:
            order_area= sum ( item.area for item in submit_list)
            s = sorted(enumerate(stock),key= lambda i:i[1].area,reverse = False)
            valid = False
            
            for S in s:
                if S[1].area>=order_area:
                    opt = submit(submit_list,stock,S[0],figObj,n)
                    valid,union = isValid(opt.GlobalBestPosition,S[1],submit_list,n)
                if valid==True:
                    o=o+submit_list
                    c=0
                    for i in opt.GlobalBestPosition:
                        p.append(i)
                        c=c+1
                        if c==n:
                            p.append(S[0])
                            c=0
                    submit_list=[]
                    stock[S[0]]=stock[S[0]].difference(union)
                    break
            if valid == False: ## an exei ftasei edw kai valid einai false simainei gia to sigekrimeno
                ## order de nborese  na vrei kanena stock na to valei 
                ## kantou tin zwi eukoli kai split sta 2
                if len(submit_list)==1:
                    print("Item couldn't fit")
                    submit_list=[]
                else:    
                    for i in range(len(submit_list)//2):
                        order.append(submit_list.pop())
            
            if not submit_list:
                if order:
                    submit_list=order.copy()
                    order=[]
            if not order and not submit_list:
                break

 
if __name__ =="__main__":
    options = ['PSO','DEGL','Nelder-Mead','L-BFGS-B','SLSQP','PatternSearch']

#    np.random.seed(2)               
    optima=options[5]
    
    times=[]
    r=[]
    R=[]
    Orders=[]
    iterations=[]
    for i in range(0,6):
        start_time = time.time()    
        r,orders,it = solution(Stock,[Order1,Order2,Order3])
        times.append(time.time() - start_time)
        Orders.append(orders)
        R.append(r)
        iterations.append(it)
        
    print("--- %s seconds ---" % (time.time() - start_time))
    #showResult(np.asarray(r),Stock,orders,4)
    showResult(np.asarray(r),Stock,orders,4)
  #  showResult(np.asarray(r),[Stock[5],Stock[6]],[Order3],4)
