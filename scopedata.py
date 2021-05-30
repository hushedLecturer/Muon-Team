# -*- coding: utf-8 -*-
"""
Oscillo-scintillator Data Processing

Created on Fri Mar 12 22:59:51 2021

@author: Hunter Louscher

Recent Changes:

    histograms of time-blocks rather than whole dataset (see accompanying Jupyter Notebook!) 
    Sun May 30 00:00:00 2021

Future Improvements:
    
    option to only take areas of "first lump"
    

"""
import numpy as np
from random import random
from matplotlib import pyplot as plt
import os
from datetime import datetime

#%% This section has everything you need to process data

#This produces an array from a csv file

def CSVToSignal(fileloc):
    '''
    This takes a file location string and returns an array.

    Parameters
    ----------
    fileloc : string
        file location, i.e. 'samplesignal.csv'.
    get_filedate : bool, optional
        if True, extracts the epoch time and adds it to the return as a tuple

    Returns
    -------
    np.array
        array of [x,y] pairs to fit the other functions "signal" parameter
        
    OR
    
    tuple
        ([same array], [epoch time])

    '''
    
    with open(fileloc) as file:
        lines = file.readlines()[11:]
        
    matrix = []
    for line in lines:
        matrix.append(line.strip().split(','))
        
    outarray = np.array(matrix).astype(np.float32)
    
    return outarray

#Gets the date from the particular files we are using as a timestamp

def dateFromFilename(fileloc):
    '''
    This takes a file location string that ends with
    waveform_ch1_[yyyy][mm][dd][hh][mm][ss][.001].csv
    and returns the epoch time.
    
    Parameters
    ----------
    fileloc : string
        file location, i.e. 'waveform_ch1_20210305211323794.csv'
        
    Returns
    float
         epoch time in seconds       
    -------
    
    '''
    year = int(fileloc[-21:-17])
    month = int(fileloc[-17:-15])
    day = int(fileloc[-15:-13])
    hour = int(fileloc[-13:-11])
    minute = int(fileloc[-11:-9])
    second = int(fileloc[-9:-7])
    milli = int(fileloc[-7:-4])
    
    #print('This signal was generated on %02d/%02d/%04d at %02d:%02d:%02d.%03d.' % (month,day,year,hour,minute,second,milli));
    
    datestring = '%04d-%02d-%02d %02d:%02d:%02d.%03d' % (year, month, day, hour, minute, second, milli)
    
    #print(datestring);
    
    dateobj = datetime.fromisoformat(datestring)
    
    sinceEp = dateobj.timestamp()
    
    #print(sinceep)

    return sinceEp

#This retrieves the voltage baseline, and the signal start and end indices

def baseAndIndex(signal):
    '''
    Finds the baseline voltage and the indices at the start and end of bumps in the signal.

    Parameters
    ----------
    signal : np.array (Nx2) 
        This is ideally an array of unitless (timestamp, voltage) pairs.
    timestamp : float, optional
        time since epoch

    Returns
    -------
    baseline : float
        The mean value for the baseline fluctuating voltage before the signal.
    signalstartindex : int
        The index once the signal diverges from the baseline by 2 std. devs. for 2 indices in a row.
    signalendindex : int
        The index found the same way, counting from the end of the signal.

    '''
    runningtotal = 0
    runningsqtotal = 0
    triggerdelay = 3
    signalstartindex = 0
    baseline = 0
    baseuncert = 0
    count = 0
    for i in range(len(signal)):
        time, value = signal[i]
        runningtotal += value
        runningsqtotal += value**2
        runningmean = runningtotal/(i+1)
        runningmeansq = runningsqtotal/(i+1)
        runninguncert = np.sqrt(runningmeansq-runningmean**2)
        if i>50 and abs(value-runningmean)/runninguncert>3:
            count+=1
        else:
            count = 0
        if count == triggerdelay:
            signalstartindex = i
            baseline = runningmean
            baseuncert = runninguncert
            break
        
    signalendindex = 0
    runningtotal = 0
    runningsqtotal = 0
    count = 0
    for i in range(len(signal)):
       time, value = signal[-i]
    
       if abs(value-baseline)/baseuncert>3:
           count+=1
       else:
           count = 0
       if count == triggerdelay:
           signalendindex = len(signal)-i
           break
    return (baseline,signalstartindex,signalendindex)

#This finds the area of the signal

def signalArea(signal,baseline=0,startindex=0,endindex=-1):
    '''
    Takes a waveform array and returns the trapezoidal area.

    Parameters
    ----------
    signal : array
        An array of (x,y) number pairs.
    baseline : float, optional
        The mean voltage before the burst. The default is 0.
    startindex : positive integer, optional
        the first index to take area over. The default is 0.
    endindex : TYPE, optional
        the last index to take area over. The default is -1.

    Returns
    -------
    area : float
        The area of the bound region by trapezoidal approximation.
    infinityError: bool
        Whether or not the waveform had to clip infinity entries.

    '''
    area = 0
    infinityError = False
    for i in range(len(signal[startindex:endindex]+1)):
        t1,x1 = signal[i+startindex]
        t2,x2 = signal[startindex+i+1]
        if abs(x1) == np.inf:
            x1 = -1
            infinityError = True
        if abs(x2) == np.inf:
            x2 = -1
            infinityError = True
        area += (t2-t1)*((x2+x1-baseline)/2)
    return (area,infinityError)

#%% This section contains ways to plot stuff

#This plots one signal array

def signalPlot(signal,saveToFile = False):
    '''
    Want a plot of your signal?

    Parameters
    ----------
    signal : np.array Nx2
        This is ideally an array of unitless (timestamp, voltage) pairs.
    saveToFile: string, optional
        If you put a filename ('signal.pdf', 'plot.png', etc),
        will save the plot there. The default will not save.

    Returns
    -------
    None.

    '''
    baseline,startindex,endindex = baseAndIndex(signal)
    area,trash= signalArea(signal,baseline,startindex,endindex)
    x,y = (signal[:,0],signal[:,1]);
    fig,ax = plt.subplots(figsize = (8,6))
    ax.set_ylim(.1,-1)
    ax.invert_yaxis()
    plt.title('Oscilloscope Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Potential (V)')
    plt.plot(x,y,'k',linewidth = 1,label = 'Signal')
    plt.plot(x[startindex],y[startindex],'yo',label = 'Start Time')
    plt.plot(x[endindex],y[endindex],'ro',label = 'End Time')
    plt.fill_between(x[startindex:endindex],baseline,y[startindex:endindex],color='blue',label = 'Area: %.2e V*s'%area)
    plt.legend()
    if saveToFile:
        plt.savefig(saveToFile)
    plt.show()

#This makes a histogram of all the csv-areas in a directory within epoch timerange

def CSVToHistogram(directory,saveToFile = False,N=250,scale = 'linear'):
    '''
    Takes a folder of waveform csv's and produces a histogram of their areas.

    Parameters
    ----------
    directory : string
        Directory string, leading up to the folder with no end slash i.e.
    N : integer, optional
        Number of histogram bins. The default is 250.
    scale : string, optional
        y-axis scale, "log" and "linear" are common. The default is 'linear'.
    saveToFile: string, optional
        If you put a filename ('signal.pdf', 'plot.png', etc),
        will save the plot there. The default will not save.

    Returns
    -------
    None.

    '''
    
    arealist = []
    areafails = 0
    for file in os.listdir(directory):
        infinityError = False
        
        if file.endswith('.csv'):
            signal = CSVToSignal(directory+'\\'+file)
            baseline,startindex,endindex=baseAndIndex(signal)
            area , infinityError = signalArea(signal,baseline,startindex,endindex)
            arealist.append(area)
        if infinityError:
            areafails +=1
    
    fig,ax = plt.subplots(figsize=(8,6))
    plt.title('Waveform Area Histogram')
    plt.ylabel('No. of Hits')
    plt.yscale(scale)
    plt.xlabel('Total Discharge (V*s)')
    ax.hist(arealist,bins=N)
    if saveToFile:
        plt.savefig(saveToFile)
    plt.show()
    print('Histogram complete, %i waveforms had infinite area.' %areafails)
    return None

#This makes a list of timestamp, area pairs for a time-dependent histogram

def CSVToTimeList(directory):
    '''
    Takes a folder of waveform csv's and produces a list
    of timestamp, area tuples

    Parameters
    ----------
    directory : string
        Directory string, leading up to the folder with no end slash i.e.
        
    Returns
    -------
    arealist, a sorted numpy array [[epoch timestamp], [area]] pairs

    '''
    
    arealist = []
    areafails = 0
    for file in os.listdir(directory):
        infinityError = False
        if file.endswith('.csv'):
            signal = CSVToSignal(directory+'\\'+file)
            baseline,startindex,endindex=baseAndIndex(signal)
            area , infinityError = signalArea(signal,baseline,startindex,endindex)
            time = dateFromFilename(file)
            arealist.append([time, area])
        if infinityError:
            areafails +=1
            
    np.array(arealist.sort())
    
    print('List complete, %i waveforms had infinite area.' %areafails)
    return arealist

#Narrowed histogram maker

def listogram(motherlist,startind,width):
    '''
    Takes a list of [epoch-timestamp, data-value] pairs and produces a histogram
    from the pairs between the start and end index

    Parameters
    ----------
    motherlist
        motherlist : np.array
            list of [epoch-timestamp, data-value] pairs
        startind : int
            start index
        width : int
            number of samples
    Returns
    -------
        None  
    '''
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(16,6))
    ax1.set_title('Waveform Area Histogram')
    ax1.set_ylabel('No. of Hits')
    ax1.set_xlabel('Voltage Absement (V*s)')
    ax1.hist(motherlist[startind:startind+width,1])
    ax1.set_xlim(-2e-8,0)
    
    #means = np.array([[(motherlist[i,0]-motherlist[0,0])/3600,np.mean(motherlist[i:i+width:1])] for i in range(len(motherlist[:-width]))])
    means = np.array([[(motherlist[i,0]-motherlist[0,0])/3600,np.mean(motherlist[i:i+width,1])] for i in range(len(motherlist)-width])
    ax2.set_title('Mean Progression Over Time: Window Width %.1fhr' %((motherlist[width,0]-motherlist[0,0])/3600))
    ax2.set_ylabel('Mean Voltage Absement (V*s)')
    ax2.set_xlabel('Time Since Start(hrs)')
    ax2.plot(means[:,0],means[:,1])
    
    
        
    return None

#%% This section just contains functions for generating mockup data

def signalGenerate():
    '''
    Produces a sample signal for testing the functions.
    
    Sometimes it fails. The while loop gives 5 attempts
    because I'm too lazy to fix it. Don't @ me.

    Returns
    -------
    signal : array of 2-tuples (x,y)

    '''
    failcount = 0
    while True:
        try:
            noiseamp = .01
            silence = 5000
            peakwid = 500
            peakht = -1*random()
            decaywid = 300
            rate = 1E10
            set1 = np.array([((i)/rate,noiseamp*random()) for i in range(int(silence))])
            set2 = np.array([((i+1)/rate+set1[-1,0],peakht*i/(decaywid) + noiseamp*random()) for i in range(int(decaywid*random()))])
            set3 = np.array([((i+1)/rate+set2[-1,0],peakht+noiseamp*random()) for i in range(int(peakwid*random()))])
            set4 = np.array([((i+1)/rate+set3[-1,0],peakht-peakht*i/decaywid + noiseamp*random()) for i in range(int(decaywid*random()))])
            set5 = np.array([((i+1)/rate+set4[-1,0],noiseamp*random()) for i in range(int(decaywid))])
            set6 = np.array([((i+1)/rate+set5[-1,0],peakht/2 - peakht*i/(2*decaywid) + noiseamp*random()) for i in range(int(decaywid*random()))])
            set7 = np.array([((i+1)/rate+set6[-1,0],noiseamp*random()) for i in range(decaywid)])
            set8 = np.array([((i+1)/rate+set7[-1,0],peakht/2 - peakht*i/(2*decaywid) + noiseamp*random()) for i in range(int(decaywid*random()))])
            set9 = np.array([((i+1)/rate+set8[-1,0],noiseamp*random()) for i in range(decaywid)])
            set10 = np.array([((i+1)/rate+set9[-1,0],peakht/4 - peakht*i/(4*decaywid) + noiseamp*random()) for i in range(int(decaywid*random()))])
            set11 = np.array([((i+1)/rate+set10[-1,0],noiseamp*random()) for i in range(int(silence/2))])
            sets = [set2,set3,set4,set5,set6,set7,set8,set9,set10,set11]
            signal = np.copy(set1)
            break
        except:
            failcount += 1
            if failcount>5:
                print('Critical failure, 5 fails in a row.')
                break
            else:
                continue
    for s in sets:
        signal = np.append(signal,s,0)
    return signal

def signalToCSV(signal,filename):
    '''
    Takes a signal and a filename, writes a file to filename, returns filename.

    Parameters
    ----------
    signal : array
        Array of (x,y) pairs from a signal.
    filename : string
        file location i.e. 'samplefile.csv'

    Returns
    -------
    filename: string

    '''
    
    with open(filename,'w') as file:
        for _ in range(20):
            file.write(',\n')
        file.write('time,value\n')
        for row in signal:
            file.write(str(row[0])+','+str(row[1])+'\n')
    print("File created at %s"%filename)    
    return filename

#%% Example usage (remove the ''' from around a chunk to use it)

# This block generates a signal array, make a file, import it,
# make an array from the file, plot it, and find the area.
'''
#make an example signal
siggy = signalGenerate()
#export it as an example file
samplefile = signalToCSV(siggy,'generatedsignal.csv')
#import the signal csv
siggy2 = CSVToSignal(samplefile)
baseline,startindex,endindex = baseAndIndex(siggy2)
print('The signal area is %.2e V*s'%signalArea(siggy2,baseline,startindex,endindex))
signalPlot(siggy2)
'''

# This block to generates 3000 sample csv's to read into a histogram
# of their areas
'''
for i in range(3000):
    mocksig = signalGenerate()
    signalToCSV(mocksig,'samples/signal%d.csv'%(i+1))

CSVToHistogram('samples')
'''
    