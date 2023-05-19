#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.1.1),
    on May 19, 2023, at 00:47
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# Run 'Before Experiment' code from code
import os
import cv2
import time
import random
import numpy as np
from numpy import inf
import matplotlib.pyplot as plt
from matplotlib import animation

# Initialize Variables
count = 0  # Track number of trials within a block
beta_hi = 0; beta_lo = 0  # Initialize beta variables
betas = [beta_hi, beta_lo]   # Initialize betas list
mistakes = 0  # Counts incorrect responses within a block
toggle = True  # True means betas converge, False they diverge

# Track Performance
performance = []
pct_correct = 0
target_beta = 0

# Converge numbers function
def converge_diverge(numbers, target, converge=True):
    converged_numbers = []  # Clear list
    
    for number in numbers:
        step_size = round(abs(target - number)/10, 3)
        #print(step_size)

        # Adjust beta_hi
        if number >= target:
            if converge:
                new_number = number - step_size
            else:
                new_number = number + step_size

        # Adjust beta_lo
        elif number <= target:
            if converge:
                new_number = number + step_size
            else:
                new_number = number - step_size

        converged_numbers.append(new_number)

    return converged_numbers

# Fractal generator class
class Generate:
    # Generate 2d or 3d fractal
    def __init__(self, beta=4, seed=117, size=256, dimension=2, preview=False, save=False, method="ifft", gs=True):
        # Set Seed
        np.random.seed(seed)
        # Set Size
        size = size+1
        # Set properties
        self.beta = beta
        self.seed = seed
        self.size = size
        self.dimension = dimension

        #Alert related
        assert self.dimension == 2 or self.dimension == 3, "Dimension must be either 2 or 3"
        np.seterr(divide='ignore')

        if dimension == 2 and method == "ifft":
            # Build power spectrum
            f = [x/size for x in range(0, int(size/2)+1)] + [x/size for x in range(-int(size/2), 0)]
            u = np.reshape(f, (size, 1))
            v = np.reshape(f, (1, size))
            powerspectrum = (u**2 + v**2)**(-beta/2)
            powerspectrum[powerspectrum == inf] = powerspectrum[0,1]
            # Noise and ifft
            phases = np.random.normal(0, 255, size=[size, size])
            pattern = np.fft.ifftn(powerspectrum**0.5 * (np.cos(2*np.pi*phases)+1j*np.sin(2*np.pi*phases)))
            # Normalize result
            pattern = np.real(pattern)
            self.pattern = (pattern-np.amin(pattern))/np.amax(pattern-np.amin(pattern))
        
        # Toggle grayscale and binarized
        if gs == False:
            self.pattern = (self.pattern > np.mean(self.pattern))  # above mean = 1, below mean = 0 

        if dimension == 3 and method == "ifft":
            # Build power spectrum
            f = np.around([x/size for x in range(0, int(size/2)+1)] + [x/size for x in range(-int(size/2), 0)], 4)
            u = np.reshape(f, (size, 1))
            v = np.reshape(f, (1, size))
            w = np.reshape(f, (size, 1, 1))
            powerspectrum = (u**2 + v**2 + w**2)**(-beta/2)
            powerspectrum[powerspectrum == inf] = powerspectrum[0,1,0]
            # Noise and ifft
            phases = np.random.normal(0, 255, size=[size, size, size])
            pattern = np.fft.ifftn(powerspectrum**0.5 * (np.cos(2*np.pi*phases)+1j*np.sin(2*np.pi*phases)))
            # Normalize result
            pattern = np.real(pattern)
            self.pattern = (pattern-np.amin(pattern))/np.amax(pattern-np.amin(pattern))

    def previewAnim(self, reps=3, mode='gs'):
        if reps == 1:
            reps = 2
        for i in range(reps-1):
            for k in range(self.size):
                cv2.imshow('Fractal Preview', self.pattern[k, :, :])
                cv2.waitKey(16)

    def preview2d(self, index=-1, size=256):
        # 2d grayscale and BW previews
        if self.dimension == 2:
            preview = cv2.resize(self.pattern, [size, size], interpolation=cv2.INTER_AREA)
            prev_bw = (preview > .5)
            previews = [preview, prev_bw]
            for i in range(2):
                plt.subplot(1, 2, i+1), plt.imshow(previews[i], 'Greys')
                plt.xticks([]), plt.yticks([])
            plt.show()
        # 2d slices of 3d fractals for preview
        if self.dimension == 3:
            if index != -1:
                assert 0 < index <= 100, "Index must be between 1-100"
                frame = int((index/100)*self.size)
            else:
                frame = -1
            preview = cv2.resize(self.pattern[frame, :, :], [size, size], interpolation=cv2.INTER_AREA)
            prev_bw = (preview > .5)
            previews = [preview, prev_bw]
            for i in range(2):
                plt.subplot(1, 2, i+1), plt.imshow(previews[i], 'Greys')
                plt.xticks([]), plt.yticks([])
            plt.show()

    def preview3d(self):
        # Check if 3 dimensional, and resize to 64x64x64
        assert self.pattern.ndim == 3, "Fractal must be 3 dimensional"
        prev3d = resize(self.pattern, (64, 64, 64))

        # Create vectors for 3d plot
        z, x, y = prev3d.nonzero()
        color = prev3d.flatten()
        color = color[:]

        #Display 3d Fractal
        fig = plt.figure()
        plt.rcParams["figure.figsize"] = 5, 5
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, c=color, alpha=1, cmap="Greys")
        plt.show()

    def boxcount(self, threshold=.5, frame=False):
        # 2d box count function
        if self.pattern.ndim == 2 or frame:
            def count(img, k):
                box = np.add.reduceat(
                    np.add.reduceat(fractal, np.arange(0, fractal.shape[0], k), axis=0),
                    np.arange(0, fractal.shape[1], k), axis=1)
                return len(np.where((box > 0) & (box < k*k))[0])

        # 3d box count function
        elif self.pattern.ndim == 3:
            def count(img, k):
                reducer = np.add.reduceat(np.add.reduceat(fractal, np.arange(0, fractal.shape[0], k), axis=0),
                                          np.arange(0, fractal.shape[1], k), axis=1)
                box = np.add.reduceat(reducer, np.arange(0, fractal.shape[2], k), axis=2)
                return len(np.where((box > 0) & (box < k*k*k))[0])

        # Threshold and box count
        fractal = (self.pattern < threshold)
        p = min(fractal.shape)
        n = 2**np.floor(np.log(p)/np.log(2))
        n = int(np.log(n)/np.log(2))
        sizes = 2**np.arange(n-1, 0, -1)
        counts = []
        for size in sizes:
            counts.append(count(fractal, size))
        m, b = np.polyfit(np.log(sizes), np.log(counts), 1)
        return -m

    def avgBoxcount(self):
        # Check if fractal is 3d
        assert self.pattern.ndim == 3, "Average box count is for 3d fractals only."

        def abc(fractal):
            def count(fractal, k):
                box = np.add.reduceat(
                    np.add.reduceat(fractal, np.arange(0, fractal.shape[0], k), axis=0),
                    np.arange(0, fractal.shape[1], k), axis=1)
                return len(np.where((box > 0) & (box < k*k))[0])

            # Threshold and box count
            fractal = (self.pattern < .5)
            p = min(fractal.shape)
            n = 2**np.floor(np.log(p)/np.log(2))
            n = int(np.log(n)/np.log(2))
            sizes = 2**np.arange(n-1, 0, -1)
            counts = []
            for size in sizes:
                counts.append(count(fractal, size))
            m, b = np.polyfit(np.log(sizes), np.log(counts), 1)
            return -m

        boxcounts = []

        for i in range(0, len(self.pattern), 3):
            frame = self.pattern[i, :, :]
            slope2d = abc(frame)
            boxcounts.append(slope2d)
        return np.mean(boxcounts)

    def write(self, location="E:/fractals"):

        # Check if root directory exists
        assert os.path.exists(location), "Root directory doesn't exist."

        # Save 2d fractal
        if self.dimension == 2:
            #folder = f"{location}/{self.seed}/{self.beta}/"
            #os.mkdir(folder)
            cv2.imwrite(f"{self.beta}_{self.seed}.png", self.pattern*255)
        if self.dimension == 3:
            folder = f"{location}/{self.seed}/{self.beta}/"
            if not os.path.exists(folder):
                os.makedirs(folder)
            for i in range(self.size):
                cv2.imwrite(f"{folder}{self.beta}_{self.seed}_{i:03d}.png", self.pattern[i, :, :]*255)


# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)
# Store info about the experiment session
psychopyVersion = '2023.1.1'
expName = 'staircase'  # from the Builder filename that created this script
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
}
# --- Show participant info dialog --
dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + u'data/test_data'

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath='E:\\Git Repos\\staircase\\staircase_lastrun.py',
    savePickle=True, saveWideText=True,
    dataFileName=filename)
# save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp
frameTolerance = 0.001  # how close to onset before 'same' frame

# Start Code - component code to be run after the window creation

# --- Setup the Window ---
win = visual.Window(
    size=[1920, 1080], fullscr=True, screen=0, 
    winType='pyglet', allowStencil=False,
    monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
    backgroundImage='', backgroundFit='none',
    blendMode='avg', useFBO=True, 
    units='height')
win.mouseVisible = False
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess
# --- Setup input devices ---
ioConfig = {}

# Setup iohub keyboard
ioConfig['Keyboard'] = dict(use_keymap='psychopy')

ioSession = '1'
if 'session' in expInfo:
    ioSession = str(expInfo['session'])
ioServer = io.launchHubServer(window=win, **ioConfig)
eyetracker = None

# create a default keyboard (e.g. to check for escape)
defaultKeyboard = keyboard.Keyboard(backend='iohub')

# --- Initialize components for Routine "introduction" ---
text = visual.TextStim(win=win, name='text',
    text='In the following experiment, you will be presented with pairs of images and asked to decide which is more complex (i.e. has finer details).\n\nWhen you have decided which image has more fine details, click directly on it to submit your answer.\n\nThe experiment will test your ability to detect and discriminate fine details across randomly generated visual stimuli.\n\nPress the space bar to begin!',
    font='Open Sans',
    pos=(0, 0), height=0.025, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
spacebar_1 = keyboard.Keyboard()

# --- Initialize components for Routine "init_beta" ---

# --- Initialize components for Routine "trial" ---
image_l = visual.ImageStim(
    win=win,
    name='image_l', 
    image='default.png', mask=None, anchor='center',
    ori=0.0, pos=(-0.33, 0), size=None,
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-1.0)
image_r = visual.ImageStim(
    win=win,
    name='image_r', 
    image='default.png', mask=None, anchor='center',
    ori=0.0, pos=(0.33, 0), size=None,
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-2.0)
mouse = event.Mouse(win=win)
x, y = [None, None]
mouse.mouseClock = core.Clock()

# --- Initialize components for Routine "next_beta" ---
pause_text = visual.TextStim(win=win, name='pause_text',
    text='Good job!\nPlease press the space bar to move to the next block.',
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
key_resp = keyboard.Keyboard()

# --- Initialize components for Routine "Debrief" ---
thank_you = visual.TextStim(win=win, name='thank_you',
    text='Thank You!\n',
    font='Open Sans',
    pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine 

# --- Prepare to start Routine "introduction" ---
continueRoutine = True
# update component parameters for each repeat
spacebar_1.keys = []
spacebar_1.rt = []
_spacebar_1_allKeys = []
# keep track of which components have finished
introductionComponents = [text, spacebar_1]
for thisComponent in introductionComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
frameN = -1

# --- Run Routine "introduction" ---
routineForceEnded = not continueRoutine
while continueRoutine:
    # get current time
    t = routineTimer.getTime()
    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *text* updates
    
    # if text is starting this frame...
    if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        text.frameNStart = frameN  # exact frame index
        text.tStart = t  # local t and not account for scr refresh
        text.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'text.started')
        # update status
        text.status = STARTED
        text.setAutoDraw(True)
    
    # if text is active this frame...
    if text.status == STARTED:
        # update params
        pass
    
    # *spacebar_1* updates
    waitOnFlip = False
    
    # if spacebar_1 is starting this frame...
    if spacebar_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        spacebar_1.frameNStart = frameN  # exact frame index
        spacebar_1.tStart = t  # local t and not account for scr refresh
        spacebar_1.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(spacebar_1, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'spacebar_1.started')
        # update status
        spacebar_1.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(spacebar_1.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(spacebar_1.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if spacebar_1.status == STARTED and not waitOnFlip:
        theseKeys = spacebar_1.getKeys(keyList=['space'], waitRelease=False)
        _spacebar_1_allKeys.extend(theseKeys)
        if len(_spacebar_1_allKeys):
            spacebar_1.keys = _spacebar_1_allKeys[-1].name  # just the last key pressed
            spacebar_1.rt = _spacebar_1_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
        if eyetracker:
            eyetracker.setConnectionState(False)
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        routineForceEnded = True
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in introductionComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# --- Ending Routine "introduction" ---
for thisComponent in introductionComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# check responses
if spacebar_1.keys in ['', [], None]:  # No response was made
    spacebar_1.keys = None
thisExp.addData('spacebar_1.keys',spacebar_1.keys)
if spacebar_1.keys != None:  # we had a response
    thisExp.addData('spacebar_1.rt', spacebar_1.rt)
thisExp.nextEntry()
# the Routine "introduction" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of conditions etc
styles = data.TrialHandler(nReps=1.0, method='random', 
    extraInfo=expInfo, originPath=-1,
    trialList=data.importConditions('styles.xlsx'),
    seed=None, name='styles')
thisExp.addLoop(styles)  # add the loop to the experiment
thisStyle = styles.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisStyle.rgb)
if thisStyle != None:
    for paramName in thisStyle:
        exec('{} = thisStyle[paramName]'.format(paramName))

for thisStyle in styles:
    currentLoop = styles
    # abbreviate parameter names if possible (e.g. rgb = thisStyle.rgb)
    if thisStyle != None:
        for paramName in thisStyle:
            exec('{} = thisStyle[paramName]'.format(paramName))
    
    # set up handler to look after randomisation of conditions etc
    beta = data.TrialHandler(nReps=1.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('target_betas.xlsx'),
        seed=None, name='beta')
    thisExp.addLoop(beta)  # add the loop to the experiment
    thisBeta = beta.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisBeta.rgb)
    if thisBeta != None:
        for paramName in thisBeta:
            exec('{} = thisBeta[paramName]'.format(paramName))
    
    for thisBeta in beta:
        currentLoop = beta
        # abbreviate parameter names if possible (e.g. rgb = thisBeta.rgb)
        if thisBeta != None:
            for paramName in thisBeta:
                exec('{} = thisBeta[paramName]'.format(paramName))
        
        # --- Prepare to start Routine "init_beta" ---
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from set_betas
        if count == 0:
            beta_lo = target_beta - 1
            beta_hi = target_beta + 1 
        # keep track of which components have finished
        init_betaComponents = []
        for thisComponent in init_betaComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "init_beta" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
                if eyetracker:
                    eyetracker.setConnectionState(False)
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in init_betaComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "init_beta" ---
        for thisComponent in init_betaComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # the Routine "init_beta" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        stimuli = data.TrialHandler(nReps=1000.0, method='random', 
            extraInfo=expInfo, originPath=-1,
            trialList=[None],
            seed=None, name='stimuli')
        thisExp.addLoop(stimuli)  # add the loop to the experiment
        thisStimulus = stimuli.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisStimulus.rgb)
        if thisStimulus != None:
            for paramName in thisStimulus:
                exec('{} = thisStimulus[paramName]'.format(paramName))
        
        for thisStimulus in stimuli:
            currentLoop = stimuli
            # abbreviate parameter names if possible (e.g. rgb = thisStimulus.rgb)
            if thisStimulus != None:
                for paramName in thisStimulus:
                    exec('{} = thisStimulus[paramName]'.format(paramName))
            
            # --- Prepare to start Routine "trial" ---
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from code
            # Set up stimuli for 2AFC
            betas = [beta_hi, beta_lo]
            random.shuffle(betas) 
            
            # In case betas are equal, separate them slightly
            if betas[0] == betas[1]:
                betas[0] = betas[0] - .01
            
            # Generate fractal patterns based on style condition
            if style == 'two_seed_gs':
                rng_l = random.randint(123, 12300)
                rng_r = random.randint(123, 12300)
                fractal_l = Generate(beta=betas[0], seed=rng_l, size=512, dimension=2)
                fractal_r = Generate(beta=betas[1], seed=rng_r, size=512, dimension=2)
            
            elif style == 'two_seed_bw':
                rng_l = random.randint(123, 12300)
                rng_r = random.randint(123, 12300)
                fractal_l = Generate(beta=betas[0], seed=rng_l, size=512, dimension=2, gs=False)
                fractal_r = Generate(beta=betas[1], seed=rng_r, size=512, dimension=2, gs=False)
            
            # Save fractals as png to load in as PsychoPy image
            cv2.imwrite(f"stimuli/fractal_left.png", fractal_l.pattern*255)
            cv2.imwrite(f"stimuli/fractal_right.png", fractal_r.pattern*255)
            
            # Define correct side
            if betas[0] < betas[1]:
                correct = image_l
            else:
                correct = image_r
            image_l.setImage('stimuli/fractal_left.png')
            image_r.setImage('stimuli/fractal_right.png')
            # setup some python lists for storing info about the mouse
            mouse.x = []
            mouse.y = []
            mouse.leftButton = []
            mouse.midButton = []
            mouse.rightButton = []
            mouse.time = []
            mouse.corr = []
            mouse.clicked_name = []
            gotValidClick = False  # until a click is received
            # keep track of which components have finished
            trialComponents = [image_l, image_r, mouse]
            for thisComponent in trialComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "trial" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *image_l* updates
                
                # if image_l is starting this frame...
                if image_l.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_l.frameNStart = frameN  # exact frame index
                    image_l.tStart = t  # local t and not account for scr refresh
                    image_l.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_l, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    image_l.status = STARTED
                    image_l.setAutoDraw(True)
                
                # if image_l is active this frame...
                if image_l.status == STARTED:
                    # update params
                    pass
                
                # if image_l is stopping this frame...
                if image_l.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_l.tStartRefresh + 1000-frameTolerance:
                        # keep track of stop time/frame for later
                        image_l.tStop = t  # not accounting for scr refresh
                        image_l.frameNStop = frameN  # exact frame index
                        # update status
                        image_l.status = FINISHED
                        image_l.setAutoDraw(False)
                
                # *image_r* updates
                
                # if image_r is starting this frame...
                if image_r.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_r.frameNStart = frameN  # exact frame index
                    image_r.tStart = t  # local t and not account for scr refresh
                    image_r.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_r, 'tStartRefresh')  # time at next scr refresh
                    # update status
                    image_r.status = STARTED
                    image_r.setAutoDraw(True)
                
                # if image_r is active this frame...
                if image_r.status == STARTED:
                    # update params
                    pass
                
                # if image_r is stopping this frame...
                if image_r.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_r.tStartRefresh + 1000-frameTolerance:
                        # keep track of stop time/frame for later
                        image_r.tStop = t  # not accounting for scr refresh
                        image_r.frameNStop = frameN  # exact frame index
                        # update status
                        image_r.status = FINISHED
                        image_r.setAutoDraw(False)
                # *mouse* updates
                
                # if mouse is starting this frame...
                if mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    mouse.frameNStart = frameN  # exact frame index
                    mouse.tStart = t  # local t and not account for scr refresh
                    mouse.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(mouse, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.addData('mouse.started', t)
                    # update status
                    mouse.status = STARTED
                    mouse.mouseClock.reset()
                    prevButtonState = mouse.getPressed()  # if button is down already this ISN'T a new click
                if mouse.status == STARTED:  # only update if started and not finished!
                    buttons = mouse.getPressed()
                    if buttons != prevButtonState:  # button state changed?
                        prevButtonState = buttons
                        if sum(buttons) > 0:  # state changed to a new click
                            # check if the mouse was inside our 'clickable' objects
                            gotValidClick = False
                            clickableList = core.getFromNames([image_l, image_r], namespace=locals())
                            for obj in clickableList:
                                # is this object clicked on?
                                if obj.contains(mouse):
                                    gotValidClick = True
                                    mouse.clicked_name.append(obj.name)
                            # check whether click was in correct object
                            if gotValidClick:
                                corr = 0
                                corrAns = core.getFromNames(correct, namespace=locals())
                                for obj in corrAns:
                                    # is this object clicked on?
                                    if obj.contains(mouse):
                                        corr = 1
                                mouse.corr.append(corr)
                            x, y = mouse.getPos()
                            mouse.x.append(x)
                            mouse.y.append(y)
                            buttons = mouse.getPressed()
                            mouse.leftButton.append(buttons[0])
                            mouse.midButton.append(buttons[1])
                            mouse.rightButton.append(buttons[2])
                            mouse.time.append(mouse.mouseClock.getTime())
                            if gotValidClick:
                                continueRoutine = False  # end routine on response
                
                # check for quit (typically the Esc key)
                if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                    core.quit()
                    if eyetracker:
                        eyetracker.setConnectionState(False)
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trialComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trial" ---
            for thisComponent in trialComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # Run 'End Routine' code from code
            count += 1
            
            # Nudge betas toward target
            converged_betas = converge_diverge(betas, target_beta, converge=toggle)
            beta_hi = np.max(converged_betas)
            beta_lo = np.min(converged_betas)
            
            # Record trial info
            thisExp.addData('beta_l', betas[0])
            thisExp.addData('beta_r', betas[1])
            thisExp.addData('seed_l', rng_l)
            thisExp.addData('seed_r', rng_r)
            thisExp.addData('style', style)
            thisExp.addData('target_beta', target_beta)
            
            # Append most recent answer to performance list 
            performance.append(mouse.corr[0])
            
            # If incorrect answer, betas diverge
            if performance[-1] == 0:
                toggle = False
                mistakes += 1
                
            # If 3x correct answers, betas converge again
            if np.sum(performance[-3:]) == 3:
                toggle = True
            
            # If 3 incorrect reponses, move to next block
            if mistakes == 3:
                count = 0  # Reset trial counter
                mistakes = 0  # reset reversals
                toggle = True
                #print(f"failure at {target_beta}")
                break # Exit stimulus loop and move to next iteration in beta_blocks loop
            
            # Log activity on console
            #if toggle == True:
            #    print("converge")
            #else:
            #    print("diverge")
            # store data for stimuli (TrialHandler)
            stimuli.addData('mouse.x', mouse.x)
            stimuli.addData('mouse.y', mouse.y)
            stimuli.addData('mouse.leftButton', mouse.leftButton)
            stimuli.addData('mouse.midButton', mouse.midButton)
            stimuli.addData('mouse.rightButton', mouse.rightButton)
            stimuli.addData('mouse.time', mouse.time)
            stimuli.addData('mouse.corr', mouse.corr)
            stimuli.addData('mouse.clicked_name', mouse.clicked_name)
            # the Routine "trial" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
        # completed 1000.0 repeats of 'stimuli'
        
        
        # --- Prepare to start Routine "next_beta" ---
        continueRoutine = True
        # update component parameters for each repeat
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
        # keep track of which components have finished
        next_betaComponents = [pause_text, key_resp]
        for thisComponent in next_betaComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "next_beta" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *pause_text* updates
            
            # if pause_text is starting this frame...
            if pause_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                pause_text.frameNStart = frameN  # exact frame index
                pause_text.tStart = t  # local t and not account for scr refresh
                pause_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(pause_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'pause_text.started')
                # update status
                pause_text.status = STARTED
                pause_text.setAutoDraw(True)
            
            # if pause_text is active this frame...
            if pause_text.status == STARTED:
                # update params
                pass
            
            # *key_resp* updates
            waitOnFlip = False
            
            # if key_resp is starting this frame...
            if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp.frameNStart = frameN  # exact frame index
                key_resp.tStart = t  # local t and not account for scr refresh
                key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp.started')
                # update status
                key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp.status == STARTED and not waitOnFlip:
                theseKeys = key_resp.getKeys(keyList=['y','n','left','right','space'], waitRelease=False)
                _key_resp_allKeys.extend(theseKeys)
                if len(_key_resp_allKeys):
                    key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                    key_resp.rt = _key_resp_allKeys[-1].rt
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
                if eyetracker:
                    eyetracker.setConnectionState(False)
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in next_betaComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "next_beta" ---
        for thisComponent in next_betaComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # check responses
        if key_resp.keys in ['', [], None]:  # No response was made
            key_resp.keys = None
        beta.addData('key_resp.keys',key_resp.keys)
        if key_resp.keys != None:  # we had a response
            beta.addData('key_resp.rt', key_resp.rt)
        # the Routine "next_beta" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
    # completed 1.0 repeats of 'beta'
    
# completed 1.0 repeats of 'styles'


# --- Prepare to start Routine "Debrief" ---
continueRoutine = True
# update component parameters for each repeat
# keep track of which components have finished
DebriefComponents = [thank_you]
for thisComponent in DebriefComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
frameN = -1

# --- Run Routine "Debrief" ---
routineForceEnded = not continueRoutine
while continueRoutine:
    # get current time
    t = routineTimer.getTime()
    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *thank_you* updates
    
    # if thank_you is starting this frame...
    if thank_you.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        thank_you.frameNStart = frameN  # exact frame index
        thank_you.tStart = t  # local t and not account for scr refresh
        thank_you.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(thank_you, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'thank_you.started')
        # update status
        thank_you.status = STARTED
        thank_you.setAutoDraw(True)
    
    # if thank_you is active this frame...
    if thank_you.status == STARTED:
        # update params
        pass
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
        if eyetracker:
            eyetracker.setConnectionState(False)
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        routineForceEnded = True
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in DebriefComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# --- Ending Routine "Debrief" ---
for thisComponent in DebriefComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# the Routine "Debrief" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# --- End experiment ---
# Flip one final time so any remaining win.callOnFlip() 
# and win.timeOnFlip() tasks get executed before quitting
win.flip()

# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename+'.csv', delim='auto')
thisExp.saveAsPickle(filename)
logging.flush()
# make sure everything is closed down
if eyetracker:
    eyetracker.setConnectionState(False)
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()
