#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 13:19:24 2019

@author: panagiotis
"""

import numpy as np
import warnings
from random import randint, random

try:
    from joblib import Parallel, delayed
    import multiprocessing
    HaveJoblib = True
except ImportError:
    HaveJoblib = False



class DEGL:
    def __init__( self
                 , ObjectiveFcn
                 , nVars
                 , LowerBounds = None
                 , UpperBounds = None
                 , D = None
                 , FunctionTolerance = 1.0e-6
                 , MaxStallIterations = 20
                 , Nf = 0.1
                 , alpha = 0.8
                 , beta = 0.8
                 , wmin = 0.4
                 , wmax = 0.8
                 , MaxIterations = None
                 , OutputFcn = None
                 , UseParallel = False
                 , Cr=0.8
                 ):
        self.ObjectiveFcn = ObjectiveFcn
        self.nVars = nVars
        self.alpha = alpha
        self.beta = beta
        self.wmin = wmin
        self.wmax = wmax
        self.Nf=Nf
        self.Cr=Cr
        if D is None:
            self.D = min(200, 10*nVars)
        else:
            assert np.isscalar(D) and D > 1, \
                "The D option must be a scalar integer greater than 1."
            self.D = max(2, int(round(self.D)))
        
        
        assert np.isscalar(FunctionTolerance) and FunctionTolerance >= 0.0, \
                "The FunctionTolerance option must be a scalar number greater or equal to 0."
        self.FunctionTolerance = FunctionTolerance
        
        if MaxIterations is None:
            self.MaxIterations = 100*nVars
        else:
            assert np.isscalar(MaxIterations), "The MaxIterations option must be a scalar integer greater than 0."
            self.MaxIterations = max(1, int(round(MaxIterations)))
        assert np.isscalar(MaxStallIterations), \
            "The MaxStallIterations option must be a scalar integer greater than 0."
        self.MaxStallIterations = max(1, int(round(MaxStallIterations)))
        
        self.OutputFcn = OutputFcn
        assert np.isscalar(UseParallel) and (isinstance(UseParallel,bool) or isinstance(UseParallel,np.bool_)), \
            "The UseParallel option must be a scalar boolean value."
        self.UseParallel = UseParallel
        
        # lower bounds
        if LowerBounds is None:
            self.LowerBounds = -1000.0 * np.ones(nVars)
        elif np.isscalar(LowerBounds):
            self.LowerBounds = LowerBounds * np.ones(nVars)
        else:
            self.LowerBounds = np.array(LowerBounds, dtype=float)
        self.LowerBounds[~np.isfinite(self.LowerBounds)] = -1000.0
        assert len(self.LowerBounds) == nVars, \
            "When providing a vector for LowerBounds its number of element must equal the number of problem variables."
        # upper bounds
        if UpperBounds is None:
            self.UpperBounds = 1000.0 * np.ones(nVars)
        elif np.isscalar(UpperBounds):
            self.UpperBounds = UpperBounds * np.ones(nVars)
        else:
            self.UpperBounds = np.array(UpperBounds, dtype=float)
        self.UpperBounds[~np.isfinite(self.UpperBounds)] = 1000.0
        assert len(self.UpperBounds) == nVars, \
            "When providing a vector for UpperBounds its number of element must equal the number of problem variables."
        
        assert np.all(self.LowerBounds <= self.UpperBounds), \
            "Upper bounds must be greater or equal to lower bounds for all variables."
        
        
        # check that we have joblib if UseParallel is True
        if self.UseParallel and not HaveJoblib:
            warnings.warn("""If UseParallel is set to True, it requires the joblib package that could not be imported; swarm objective values will be computed in serial mode instead.""")
            self.UseParallel = False
        
        
        lbMatrix = np.tile(self.LowerBounds, (self.D, 1)) 
        ubMatrix = np.tile(self.UpperBounds, (self.D, 1))
        bRangeMatrix = ubMatrix - lbMatrix
        self.z = lbMatrix +np.random.rand(self.D,nVars)*bRangeMatrix
        self.k = np.floor(self.D*self.Nf)
        
        self.CurrentGenFitness = np.zeros (self.D)
        self.u=self.z.copy()#isws borei na 
        self.__evaluateGenerationFitness()
        
        self.PreviousBestPosition = self.z.copy()
        self.PreviousBestFitness = self.CurrentGenFitness.copy()
                
        bInd = self.CurrentGenFitness.argmin()
        self.GlobalBestFitness = self.CurrentGenFitness[bInd].copy()
        self.GlobalBestPosition = self.PreviousBestPosition[bInd, :].copy()
        
        self.Iteration = -1;
        
        self.StallCounter = 0;
        
        self.GlobalBestSoFarFitnesses = np.zeros(self.MaxIterations+1)
        self.GlobalBestSoFarFitnesses.fill(np.nan)
        self.GlobalBestSoFarFitnesses[0] = self.GlobalBestFitness
                
        if self.OutputFcn:
            self.OutputFcn(self)
            
    def __evaluateGenerationFitness(self):
        if self.UseParallel:
            nCores = multiprocessing.cpu_count()
            self.CurrentGenFitness[:] = Parallel(n_jobs=nCores)( 
                    delayed(self.ObjectiveFcn)(self.u[i,:]) 
                    for i in range(self.D) )
        else:
            self.CurrentGenFitness[:] = [self.ObjectiveFcn(self.u[i,:]) for i in range(self.D)]
    
    def optimize(self):
        nVars = self.nVars
        k = self.k
        D = self.D
        L =np.zeros([D,nVars])
        y =np.zeros([D,nVars])
        g =np.zeros([D,nVars])
        self.u = np.zeros([D,nVars])
        stop = False
      
        while not stop:
            self.Iteration+=1 #iteration starts from 0 
            
            w = self.wmin + (self.wmax-self.wmin )*(self.Iteration)/(self.MaxIterations-1)
            for i in range(D):
               # L = self.z + 
                neigh=np.array([(v+D if v<0 else v if v<D else v-D)
                             for v in range(i-int(k),i+int(k)+1)])
                r = np.random.choice(neigh,size=3,replace=False)
                r[r==i]=r[2]
                p = r[0]
                q = r[1]
                
                bestLocation = self.PreviousBestFitness[neigh].argmin()
                zbest = self.z[bestLocation]
                L[i,:] = self.z[i,:] + self.alpha * ( zbest - self.z[i,:] ) + self.beta * ( self.z[p,:] - self.z[q,:] ) 
                
                
                r = np.random.choice(neigh,size=3,replace=False)
                r[r==i]=r[2]
                r1 = r[0]
                r2 = r[1]
                
                bestLocation = self.PreviousBestFitness.argmin()
                zbest = self.z[bestLocation]
                
                g[i,:] = self.z[i,:] + self.alpha * ( zbest - self.z[i,:] ) + self.beta * ( self.z[r1,:] - self.z[r2,:] )
                y[i,:] = w * g[i,:] + (1 - w) * L[i,:]
                
                # Cross Over Calculation 
                
                Cr = self.Cr
                jrand= randint(0,nVars)
                for l in range(0,self.nVars):
                    if(random()<Cr or l==jrand):
                        self.u[i,l]=y[i,l]
                        
                    else:
                        self.u[i,l]=self.z[i,l]
                    
                    
                inval = self.u[i,:] < self.LowerBounds
                self.u[i,inval] = self.LowerBounds[inval]
                
                inval = self.u[i,:] > self.UpperBounds
                self.u[i,inval] = self.UpperBounds[inval]
                
            # Get CurrentGenFitness f(u)
            self.__evaluateGenerationFitness()
            #check which values have to change f(u)<f(z)
            genProgressed = self.CurrentGenFitness < self.PreviousBestFitness
            self.PreviousBestPosition[genProgressed, :] = self.u[genProgressed, :]
            self.z[genProgressed, :] = self.u[genProgressed, :]
            self.PreviousBestFitness[genProgressed] = self.CurrentGenFitness[genProgressed]
            
            
            # update global best, adaptive neighborhood size and stall counter
            newBestInd = self.CurrentGenFitness.argmin()
            newBestFit = self.CurrentGenFitness[newBestInd]
            
            if newBestFit < self.GlobalBestFitness:
                self.GlobalBestFitness = newBestFit
                self.GlobalBestPosition = self.z[newBestInd, :].copy()
                
                self.StallCounter = max(0, self.StallCounter-1)
                # calculate remaining only once when fitness is improved to save some time
                # useful for the plots created
               
            else:
                self.StallCounter += 1
                
            self.GlobalBestSoFarFitnesses[self.Iteration+1] = self.GlobalBestFitness
            
            if self.OutputFcn and self.OutputFcn(self):
                self.StopReason = 'OutputFcn requested to stop.'
                stop = True
                continue
            
            if self.Iteration >= self.MaxIterations-1:
                self.StopReason = 'MaxIterations reached.'
                stop = True
                continue
            
            if self.Iteration > self.MaxStallIterations:
                minBestFitness = self.GlobalBestSoFarFitnesses[self.Iteration+1]
                maxPastBestFit = self.GlobalBestSoFarFitnesses[self.Iteration+1-self.MaxStallIterations]
                if (maxPastBestFit == 0.0) and (minBestFitness < maxPastBestFit):
                    windowProgress = np.inf  # don't stop
                elif (maxPastBestFit == 0.0) and (minBestFitness == 0.0):
                    windowProgress = 0.0  # not progressed
                else:
                    windowProgress = abs(minBestFitness - maxPastBestFit) / abs(maxPastBestFit)
                if windowProgress <= self.FunctionTolerance:
                    self.StopReason = 'Population did not improve significantly the last MaxStallIterations.'
                    stop = True
            
        
        # print stop message
        print('Algorithm stopped after {} iterations. Best fitness attained: {}'.format(
                self.Iteration+1,self.GlobalBestFitness))
        print(f'Stop reason: {self.StopReason}')
    