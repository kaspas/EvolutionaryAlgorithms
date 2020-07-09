#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 15:31:29 2020

@author: panagiotis
"""

import math
import numpy as np
from shapely.ops import cascaded_union
import shapely
import sys


def ObjectiveFcn2(p, stock, Order,A,B,C,D,E,numberOfVar):
    '''
    p = particle of Evolutionary algorithm
    stock = the stock available
    Order = the order which has to be evaluated
    A = fsm weight
    B = close to 0,0 distance weight of objects with respect to their area 
    C = out of boundaries weight 
    D = objects overlap weight
    E = object compactness wieght
    
    '''
    n=numberOfVar
    remaining = stock
    b = stock.bounds
    orders = [shapely.affinity.rotate(Order[i],p[i*n+2],origin = 'centroid') for i in range(len(Order))]
    orders = [shapely.affinity.translate(orders[i],xoff=p[i*n + 0],yoff=p[i*n+1]) for i in range(len(Order))]
    objdis=0
    for order in orders:
        dis=0
        for order2 in orders: ## gia 2 idia antikeimena i apostasi tou kentrou einai 0 gi auto afinw na ginei i praks
            dis=dis+math.sqrt((order.centroid.x-order2.centroid.x)**2+(order.centroid.y-order2.centroid.y)**2)
        objdis=objdis+dis


#    if stock.geom_type == "Polygon":
#        dis = sum(min( math.sqrt((c[0]-orders[i].centroid.x)**2+(c[1]-orders[i].centroid.y)**2) 
#                    for c in stock.exterior.coords)*orders[i].area for i in range(len(orders)))
#    elif stock.geom_type == "MultiPolygon":
#        dis=0
#        for poly in stock:
#            for order in orders:
#                if poly.contains(order):
#                    dis = dis +min( math.sqrt((c[0]-order.centroid.x)**2+(c[1]-order.centroid.y)**2) 
#                    for c in poly.exterior.coords)*order.area
#    
    dis = sum( math.sqrt( (orders[i].centroid.x - b[0])* (orders[i].centroid.x - b[0])+
                         (orders[i].centroid.y-b[1])*(orders[i].centroid.y-b[1]))*orders[i].area for i in range(len(orders)) )
    
    
#    rotation = sum(int(p[i*n+2]%90)*orders[i].area for i in range(len(orders)))

    new = shapely.ops.cascaded_union(orders)
    if(new.is_valid==False):
        new.buffer(0)
    if(new.is_empty==True):
        new.buffer(0)
    remaining = remaining.difference(new)
    #remaining = remaining.buffer(-0.5,join_style = shapely.geometry.JOIN_STYLE.mitre)
    #remaining = remaining.buffer(0.5,join_style = shapely.geometry.JOIN_STYLE.mitre)
    
    
    outOfBounds = new.difference(stock).area
    
    shapesArea = sum(shape.area for shape in orders)
    overlap = shapesArea/new.area -1 #>=0

    ch = (remaining.convex_hull)
    if remaining.area==0:
        l =sys.float_info.max - 1
    else:
        l = (ch.area)/(remaining.area) - 1
    alpha = 1.11
    fsm = 1/(1 + alpha*l)
    

    return A*fsm+B*dis+C*outOfBounds+D*overlap+E*objdis#,remaining



def ObjectiveFcn4(p, stock, Order,A,B,C,D,E,nVars):
    '''
    p = particle of Evolutionary algorithm
    stock = the stock available either a list of shapely polygon or just a polygon
    Order = the order which has to be evaluated
    A = fsm weight
    B = weight close to 0,0 distance of object with respect to its area 
    C = out of boundaries weight 
    D = objects overlap weight
    E = rotation weights tend to go 0 if rotation is one of commonly used rotation 0 30 45 60 90
    
    '''
    remaining = stock
    b = stock.bounds
    orders = [shapely.affinity.rotate(Order[i],p[i*nVars+2],origin = 'centroid') for i in range(len(Order))]
    orders = [shapely.affinity.translate(orders[i],xoff=p[i*nVars + 0],yoff=p[i*nVars+1]) for i in range(len(Order))]
    
    dis = sum( math.sqrt( (orders[i].centroid.x - b[0])* (orders[i].centroid.x - b[0])+
    (orders[i].centroid.y-b[1])*(orders[i].centroid.y-b[1]))*orders[i].area for i in range(len(orders)) )
    
    rotation = sum(int(p[i*nVars+2]%90)*orders[i].area for i in range(len(orders)))
    bo = [orders[i].bounds for i in range(len(orders))]
    new = shapely.ops.cascaded_union(orders)
    remaining = remaining.difference(new)
#    remaining = remaining.buffer(-0.3,join_style = shapely.geometry.JOIN_STYLE.mitre)
#    remaining = remaining.buffer(0.3,join_style = shapely.geometry.JOIN_STYLE.mitre)
    outOfBounds = new.difference(stock).area
    shapesArea = sum(shape.area for shape in orders)
    overlap = shapesArea/new.area -1 #>=1
    #dis = sum(max(bo[i])*orders[i].area for i in range(len(bo)))
    #dis =sum( ((bo[i][0]-b[0])+(bo[i][1]-b[1])+(bo[i][2]-b[0])+(bo[i][3]-b[1]))*orders[i].area for i in range(len(bo))) 
    ch = (remaining.convex_hull)
    #l = (ch.area)/(remaining.area) - 1
    if remaining.area==0:
        l =sys.float_info.max - 1
    else:
        l = (ch.area)/(remaining.area) - 1
    alpha = 1.11
    fsm = 1/(1 + alpha*l)

    return A*fsm+B*dis+C*outOfBounds+D*overlap+E*rotation,remaining
    #return 1*fsm+10*dis+1000*outOfBounds+10000*overlap+rotation

def ObjectiveFcn3(p,Stock,Order,A,B,C,D,E,F,numberOfVar):
    result=0.0
    if isinstance(Stock,list):
        rem=[]
        for l in range(len(Stock)):
            P=[]
            o=[]
            
            
            for i in range(len(Order)):
                if int(p[i*numberOfVar+3])==l:
                    o.append(Order[i])
                    for j in range(numberOfVar-1):
                        P.append(p[i*numberOfVar+j])
            
            if P:
                P=np.asarray(P)            
                r,remaining=ObjectiveFcn2(P,Stock[l],o,A,B,C,D,E,numberOfVar-1)
                rem.append(remaining.area/Stock[l].area*F)
                
                result=r+result
        emptyness = sum(rem)
        result=result+emptyness
    else:
        result=ObjectiveFcn2(p,Stock,Order,A,B,C,D,E,numberOfVar)
    
    return result
            
        
