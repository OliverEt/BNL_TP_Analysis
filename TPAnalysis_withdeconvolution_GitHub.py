#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 14:57:11 2020

@author: ollie
"""


import matplotlib.pyplot as plt
import numpy as np
import cv2 #This is the OpenCV package for image recognition in python. Install for anaconda by using 'conda install -c conda-forge opencv'
from scipy import signal as sig
import mysql.connector
import pandas as pd
import os
from skimage.filters import threshold_otsu
from scipy import special as sp


#####################################
## Variables for preceeding script ##

chop=10  # defines the number of pixels near the zero point to ignore.

# Y pixel values of data
y_data_min = 195
y_data_max = 355

# Y pixel values of background
y_bg_min = 5#400
y_bg_max = 175#510

# Hot pixel removal filter level - above this level (uint16) the signal is replaced by image median
hot_pix_filter_level = (2**16) * 0.5

# Other variables
noise_sigma_level = 5
bin_size = 4

# TP camera variables
TPumperpx = 80  # camera um per pixels
zeropoint = 31  # pixel value of zero point
TP_imaging_solid_angle = 0.28472434132786645
TP_gain = 0
Cam_QE = 0.7  # for 430nm

# TP slit solid angle
a = 1e-2  # slit height
b = 600e-6  # slit width
d = 0.15  # distance to slit from interaction
solid_angle = 4 * np.arctan((a*b)/(2 * d * np.sqrt((4 * d**2)+a**2 +b**2)))  # solid angle tended by the slit/pinhole

# Plotting variables
E_max = 4  # peak energy of the energy plot

#####################################
## Make save directories ##
save_directories = ['/Users/ollie/Desktop/BNL_2020/TP/Cleaned_Images',
                    '/Users/ollie/Desktop/BNL_2020/TP/dNdE',
                    '/Users/ollie/Desktop/BNL_2020/TP/dNdE_percent',
                    '/Users/ollie/Desktop/BNL_2020/TP/Spectrum_Image',
                    '/Users/ollie/Desktop/BNL_2020/TP/Detection_Limit_CrossingPoint',
                    '/Users/ollie/Desktop/BNL_2020/TP/Peak_Index',
                    '/Users/ollie/Desktop/BNL_2020/TP/Deconvolution',
                    '/Users/ollie/Desktop/BNL_2020/TP/Deconvolution/Difference',
                    '/Users/ollie/Desktop/BNL_2020/TP/Deconvolution/Peak_Height',
                    '/Users/ollie/Desktop/BNL_2020/TP/Deconvolution/Deconvolution_Effect']

for l in np.arange(0,np.shape(save_directories)[0]):
    if not os.path.exists(save_directories[l]):
        os.makedirs(save_directories[l])
        print("Directory " , save_directories[l] ,  " Created ")
    else:    
        print("Directory " , save_directories[l] ,  " already exists")    

#####################################
## Get All TP Data Shot Filepaths ##
base_filepath = '/Volumes/ICL_IONS 1/BNL_2020_02/Data/'
os.chdir(base_filepath)

filenum = []
filepath = []
for dirpath, dirnames, filenames in os.walk("."):
#    for dirname in [q for q in dirnames if q.endswith("data_",0,4)]:
    for dirname in [d for d in dirnames if d.startswith("TP_")]:
        for filename in [f for f in filenames if f.startswith("shot") & f.endswith(".tif")]:
            find_filenum = int(filename[filename.find('0'):filename.find('.tif')])
            filenum.append(find_filenum)
            current_filepath = os.path.join(base_filepath,dirpath[2::],filename)
            filepath.append(current_filepath)
            print(os.path.join(base_filepath,dirpath[2::], filename))
            

#####################################
## Get LSF for TP Deconvolution ##
## Get Slit Function for Deconvolution ## 
def Pearson1d(x=0, A=13.6682861, sig_x=6.52918482, mu_x=74.0000000, m=1.40677814):
    
    return (A / (sig_x * sp.beta(m - 0.5,0.5))) * (1 + ((x - mu_x)/sig_x)**2)**(-m); 



x_lsf = np.linspace(0, 150, 300)
lsf = Pearson1d(x_lsf)


#####################################
## Function for TP Analysis ##
Cutoff_Energies = []
Peak_Energies = []
Total_BeamEnergy = []
Total_BeamCharge = []
Peak_Signal = []

def TP_Analysis_Func( image_path_var, shot_num):
    
    #####################################
    ## Read in the image ##
    image = cv2.imread(image_path_var,-1) # reads the image
    
    #####################################
    ## Pre-filter to remove hot pixels
# =============================================================================
#     image_pre_filter = np.where(image>hot_pix_filter_level, np.median(image),image)
# =============================================================================
    global_thresh_otsu = threshold_otsu(image[y_bg_min:y_bg_max,:])
    binary_global_otsu = image > global_thresh_otsu
    image_pre_filter = np.where(binary_global_otsu==True, np.median(image),image)
    
    ## Apply a median filter to remove hot pixels ##
    
    kernal = 9 # the dimension of the x and y axis of the filter kernal.
    
    median_image = sig.medfilt2d(input=image_pre_filter,kernel_size=kernal)
    
    #median_image = image_pre_filter
    
    filter_comparison_fig = plt.figure(figsize=(11,6))
    
    plt.suptitle("Shot " + repr(shot_num))
    
    plt.subplot(121)
    plt.imshow(image),plt.title('Original')
    plt.plot([1,image.shape[1]-1],[y_data_min, y_data_min],color='r',LineStyle='--')
    plt.plot([1,image.shape[1]-1],[y_data_max, y_data_max],color='r',LineStyle='--')
    plt.plot([1,image.shape[1]-1],[y_bg_min, y_bg_min],color='g',LineStyle='--')
    plt.plot([1,image.shape[1]-1],[y_bg_max, y_bg_max],color='g',LineStyle='--')
    
    plt.xticks([]), plt.yticks([])
    
    cbar = plt.colorbar()
    cbar.solids.set_edgecolor("face")
    plt.draw()
    
    plt.subplot(122) 
    plt.imshow(np.log10(median_image)),plt.title('Otsu + Median Filter')
    
    plt.xticks([]), plt.yticks([])
    
    cbar = plt.colorbar()
    cbar.solids.set_edgecolor("face")
    plt.draw()
    
    #plt.show()

    filter_comparison_fig.savefig('/Users/ollie/Desktop/BNL_2020/TP/Cleaned_Images/shot' + repr(shot_num) +'_cleanedImageComparison.pdf', format='pdf', dpi=300)
    plt.close()
    
    #####################################
    ## Get the background subtracted raw spectrum image and background ##
    
    raw_signal=median_image[y_data_min:y_data_max,:]  # creates array of data in band of y values signal lies within.
    background=median_image[y_bg_min:y_bg_max,:]  # creates backgroud array in band of y values defined in the range above.
    
    BGsubbed_raw_signal = np.subtract(raw_signal,np.median(background,0)) # Subtract the median of the BG for each column from the data
    
    #summed_signal = np.mean(BGsubbed_raw_signal,axis=0)
    summed_signal = np.sum(BGsubbed_raw_signal,axis=0)  # Sum over all columns to get a singal line for the data region
    noise = noise_sigma_level * np.shape(BGsubbed_raw_signal)[0] * np.std(background,axis=0) #/ np.sqrt(np.shape(background)[0]) # Get the noise level for data - multiply by number of rows in signal  
    #noise=noise*np.shape(BGsubbed_raw_signal)[0]
    
    ##########################
    ## Deconvolution ##
    final_deconv, peak_val = deconvolution(summed_signal,shot_num)
    summed_signal = final_deconv
    noise = noise[0:-1]
    
    Peak_Signal.append(peak_val[-1])
    ##########################
    
    x_min_signal = TPumperpx*1e-3
    x_max_signal = (np.shape(summed_signal)[0] - zeropoint)*TPumperpx*1e-3
    xaxis = np.linspace(x_min_signal, x_max_signal, num=np.shape(summed_signal)[0]-zeropoint-1)
    
    Eaxis = 28390*xaxis**(-5) - 2605*xaxis**(-3) + 715.5*xaxis**(-2) - 21.61*xaxis**(-1) + 0.004760*xaxis  # Spatial calibration of BNL TP dispersion
    
    
    ## Select the image regions for signal and BG from zeropoint to right hand (low energy) edge
    signal_from_zeropoint = summed_signal[zeropoint:-1]
    noise_from_zeropoint = noise[zeropoint:-1]
    
    

# =============================================================================
#     bin_number=np.round(np.shape(Eaxis)[0]/bin_size)
#      
#     binEaxis_var = []
#     binnewlineaverage_var = []
#     binnewdetect_limit_var = []
#     for x in range(0,np.int(bin_number)-1):
#         binEaxis=np.mean(Eaxis[(x*bin_size-(bin_size-1)):(bin_size*x)])  # caulculate energy of each bin
#         binnewlineaverage=np.sum(signal_from_zeropoint[(x*bin_size-(bin_size-1)):(x*bin_size)])  # signal level for each bin
#         binnewdetect_limit=np.sum(noise_from_zeropoint[(x*bin_size-(bin_size-1)):(x*bin_size)])  # detection limit for each bin
#     
#         binEaxis_var.append(binEaxis)
#         binnewlineaverage_var.append(binnewlineaverage)
#         binnewdetect_limit_var.append(binnewdetect_limit)
#      
#     Eaxis=np.asarray(binEaxis_var)
#     signal_from_zeropoint=np.asarray(binnewlineaverage_var)
#     noise_from_zeropoint=np.asarray(binnewdetect_limit_var)
# =============================================================================

    
    #####################################
    ## convert from signal to number of protons according to formula ##
    gain_func = ((-0.08145*TP_gain) + 31.9)/(TP_gain + 6.787)
    lens_tranmission=0.8
    f_capture=TP_imaging_solid_angle*lens_tranmission/(2*np.pi)
    
    n_gamma_signal = signal_from_zeropoint/(Cam_QE * gain_func)
    n_gamma_noise = noise_from_zeropoint/(Cam_QE * gain_func)
    
    gamma_signal=n_gamma_signal/f_capture
    gamma_noise=n_gamma_noise/f_capture
    
    Y=3.75708*Eaxis**(1.99482)
    
    new_signal=gamma_signal/Y
    new_noise=gamma_noise/Y
    
    #####################################
    ## Get the bin width from energy axis ##
    bin_width=[]
    for i in range(0,np.shape(Eaxis)[0]-2):
        bwidth = np.abs((Eaxis[i+2] - Eaxis[i])/2)
     #   bwidth = np.abs((Eaxis[i+2] - Eaxis[i+1])/2) + np.abs((Eaxis[i+1] - Eaxis[i])/2)
        bin_width.append(bwidth)
    
    #####################################
    ## Get final energy and data arrays ##
    
    finaldNdE = []
    final_detect_limit = []
    finaldNdE_percent = []
    final_detect_limit_percent = []
    for j in range(0,np.shape(bin_width)[0]):
        dNdE = np.abs(new_signal[j+1] / bin_width[j])
        detect_limit = np.abs(new_noise[j+1] / bin_width[j])
        
        dNdE_percent = np.abs(new_signal[j+1] / ((bin_width[j] / Eaxis[j+1])))
        detect_limit_percent = np.abs(new_noise[j+1] / ((bin_width[j] / Eaxis[j+1])))
        
        finaldNdE.append(dNdE)
        final_detect_limit.append(detect_limit)
        
        finaldNdE_percent.append(dNdE_percent)
        final_detect_limit_percent.append(detect_limit_percent)
    
    final_E_axis = Eaxis[1:np.shape(Eaxis)[0]-1]
    
    ## Correct for solid angle ##
    finaldNdE = np.array(finaldNdE) / solid_angle
    final_detect_limit = np.array(final_detect_limit) / solid_angle
    finaldNdE_percent = np.array(finaldNdE_percent) / solid_angle
    final_detect_limit_percent = np.array(final_detect_limit_percent) / solid_angle
    
    #####################################
    ## Plot ##
    
    ## dN/dE plotting ##
    dNdE_fig = plt.figure()
    plt.plot(final_E_axis,finaldNdE,Label="Signal")
    plt.plot(final_E_axis,final_detect_limit,Label="Detection Limit - " + repr(noise_sigma_level) + "$\sigma$")
    
    plt.ylabel(r'$\frac{dN}{dEd\Omega} [\# MeV^{-1} Sr^{-1}]$')
    plt.xlabel("Proton Energy [MeV]")
    plt.legend()
    
    plt.yscale('log')
    plt.ylim(1e8,1e16)
    plt.xlim(0,E_max)
    
    plt.title("Shot " + repr(shot_num))

    dNdE_fig.savefig('/Users/ollie/Desktop/BNL_2020/TP/dNdE/shot' + repr(shot_num) +'dNdE.pdf', format='pdf', dpi=300)
    plt.close()

    
    ## dN/1%dE plotting ##
    dNdE_percent_fig = plt.figure()
    plt.plot(final_E_axis,finaldNdE_percent,Label="Signal")
    plt.plot(final_E_axis,final_detect_limit_percent,Label="Detection Limit - " + repr(noise_sigma_level) + "$\sigma$")
    
    plt.ylabel(r'$\frac{dN}{dE_{0.1\%}d\Omega} [\# \mathrm{bandwidth}^{-1} Sr^{-1}]$')
    plt.xlabel("Proton Energy [MeV]")
    plt.legend()
    
    plt.yscale('log')
    plt.ylim(1e8,1e16)
    plt.xlim(0,E_max)
    
    plt.title("Shot " + repr(shot_num))
    
    dNdE_percent_fig.savefig('/Users/ollie/Desktop/BNL_2020/TP/dNdE_percent/shot' + repr(shot_num) +'dNdE_percent.pdf', format='pdf', dpi=300)
    plt.close()

    
    #####################################
    ## Display image with energy markers on axis ##
    tick_energies = ['Inf',0.2, 0.3, 0.5, 1, 2,3]
    
    energy_tick_indicies = [zeropoint]
    for j in tick_energies[1:-1]:
        indexes = np.argmin(np.abs(Eaxis - j)) + zeropoint
        energy_tick_indicies.append(indexes)
    
    cropped_spectrum_fig = plt.figure()    
    plt.imshow(np.log10(raw_signal))
    plt.xticks(ticks=energy_tick_indicies,labels=tick_energies)
    plt.xlabel('Proton Energy [MeV]')
    plt.yticks([])
    plt.title("Shot " + repr(shot_num))
    
    
    cbar = plt.colorbar()
    cbar.solids.set_edgecolor("face")
    cbar.ax.set_ylabel('$Log_{10}(Counts)$')
    plt.draw()
    
    #plt.show()
    
    cropped_spectrum_fig.savefig('/Users/ollie/Desktop/BNL_2020/TP/Spectrum_Image/shot' + repr(shot_num) +'_croppedSpectrum.pdf', format='pdf', dpi=300)
    plt.close()
    
    #####################################
    ## Get energy cut-off ##
    
    idx = []
    idx_end = []
    peak_index = []
            
    if Eaxis[np.where((((finaldNdE - final_detect_limit))>0))].size > 0:
        signchange = ((np.roll(np.sign(finaldNdE - final_detect_limit), 1) - np.sign(finaldNdE - final_detect_limit)) != 0).astype(int)
        crossover_idx = [i for i, x in enumerate(signchange) if x]
        if len(crossover_idx) < 20:
            skip = np.zeros(len(crossover_idx))
            for l in np.arange(len(crossover_idx)-1):
                if skip[l] == 0:
                    if (crossover_idx[l+1] - crossover_idx[l]) < 11:
                        skip[l+1] = 1            
                    else:
                        idx = crossover_idx[l]
                        idx_end = crossover_idx[l+1]
                        break
        
            if idx == []:
                Cutoff_Energies.append(0)
                Peak_Energies.append(0)
                
            else:
                Cutoff_Energy = Eaxis[idx]   
                Cutoff_Energies.append(np.float(Cutoff_Energy))
                
                peak_index = sig.argrelmax(summed_signal[idx+zeropoint+1:idx_end+zeropoint+1], order=5)
                if np.count_nonzero(peak_index) == 0:
                    Peak_Energies.append(0)
                else:
#                Peak_Energy = Eaxis[sig.find_peaks(summed_signal[idx+zeropoint+1:idx_end+zeropoint+1],width = 5, height = np.max(summed_signal[idx+zeropoint+1:idx_end+zeropoint+1])/3)[0][0]+idx]
                    Peak_Energy = Eaxis[peak_index[0][0]+idx]
                    Peak_Energies.append(np.float(Peak_Energy))
                    
        else:
            Cutoff_Energies.append(0)
            Peak_Energies.append(0)
    else:
        Cutoff_Energies.append(0)
        Peak_Energies.append(0)
    
    

        
        
    detect_limit = plt.figure()
    plt.plot(finaldNdE - final_detect_limit)
    plt.yscale('log')
    plt.ylim(1e-10,1e15)
    plt.title("Shot " + repr(shot_num))
    
    detect_limit.savefig('/Users/ollie/Desktop/BNL_2020/TP/Detection_Limit_CrossingPoint/shot' + repr(shot_num) +'detectionLimitCrossoverPoints.pdf', format='pdf', dpi=300)
    plt.close()
    
    
    
    
    if np.count_nonzero(peak_index) != 0:
        peak_index_fig = plt.figure()
        plt.plot(final_E_axis,finaldNdE,Label="Signal")
        plt.plot(final_E_axis,final_detect_limit,Label="Detection Limit - " + repr(noise_sigma_level) + "$\sigma$")
        plt.plot(Peak_Energies[-1], finaldNdE[peak_index[0][0]+idx], "o",fillstyle='none',Label='Peak Energy - '+ repr(float("{:.3f}".format(Peak_Energies[-1])))+'MeV')
        
        plt.ylabel(r'$\frac{dN}{dEd\Omega} [\# MeV^{-1} Sr^{-1}]$')
        plt.xlabel("Proton Energy [MeV]")
        plt.legend()
        
        plt.yscale('log')
        plt.ylim(1e8,1e16)
        plt.xlim(0,E_max)
        
        plt.title("Shot " + repr(shot_num))
    
        peak_index_fig.savefig('/Users/ollie/Desktop/BNL_2020/TP/Peak_Index/shot' + repr(shot_num) +'PeakIndex.pdf', format='pdf', dpi=300)
        plt.close()
   
    
    #####################################
    ## Get proton beam charge and energy in the proton beam for conversion efficiency ##
    
    # Beam charge
    
    # Total beam energy
    total_beam_energy = np.sum(finaldNdE[np.asarray(np.where((finaldNdE[100::] - final_detect_limit[100::])>0))+100] * Eaxis[np.asarray(np.where((finaldNdE[100::] - final_detect_limit[100::])>0))+100])*1e6 * 1.6e-19*solid_angle  # times by solid angle for measured energy in beam    
    Total_BeamEnergy.append(total_beam_energy)
    
    total_beam_charge = np.sum(finaldNdE[np.asarray(np.where((finaldNdE[100::] - final_detect_limit[100::])>0))+100]) * 1.6e-19*solid_angle
    Total_BeamCharge.append(total_beam_charge)
    
    return Cutoff_Energies, Peak_Energies, Total_BeamEnergy, Total_BeamCharge, Peak_Signal;








    #####################################
    ## Deconvolution process ## 
def deconvolution(original_signal, shot_num):
    len_diff = len(original_signal) - len(lsf)
    win = np.pad(lsf/np.sum(lsf), (int((len_diff+len(lsf))/2 - np.argmax(lsf)),int(len_diff - ((len_diff+len(lsf))/2 - np.argmax(lsf)))), mode='constant')[0:-1]
    
    original_signal = original_signal[0:-1]
    original_signal[original_signal < 0] = 0
    
    threshold = 0.001
    Nmax = 250
    data_length = np.shape(original_signal)[0]
    first_guess = original_signal
    last_guess = first_guess
    difference = np.zeros((Nmax))
    next_guess = np.zeros((data_length))
    peak_value = []
     
    for i in np.arange(0,Nmax):
        conv_part_1 = (original_signal / (sig.convolve(last_guess, win, 'same')))
        conv_part_1[np.isnan(conv_part_1)] = 0
        next_guess = last_guess * (sig.convolve(conv_part_1,np.flipud(win), 'same'))
        if i == 0:
            last_guess = next_guess
    
        else:
            diff = next_guess - last_guess
            next_guess[diff<threshold] = np.nan
            next_guess[np.isnan(next_guess)] = last_guess[np.isnan(next_guess)]
            last_guess = next_guess
            
         
        data_check = sig.convolve(win, last_guess, 'same') 
       
        difference[i] = np.sum((data_check - original_signal))  # removed the random **2 factor...
        
        peak = np.max(last_guess)
        peak_value.append(peak)
        
        
    
    
    deconvolve_effect_fig = plt.figure()
    plt.plot(original_signal,label='Original Spectrum')
    plt.plot(last_guess,label='Deconvolved Spectrum')
    plt.plot(win*10*np.max(last_guess),label = 'Arb. Scaled Slit Function')
    plt.xlabel('Pixel')
    plt.ylabel('Counts')
    plt.legend()
    plt.title("Shot " + repr(shot_num))
    
    deconvolve_effect_fig.savefig('/Users/ollie/Desktop/BNL_2020/TP/Deconvolution/Deconvolution_Effect/shot' + repr(shot_num) +'_Deconvolve_Effect.pdf', format='pdf', dpi=300)
    plt.close()
    
    diff_fig = plt.figure()
    plt.plot(difference)
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Last_Guess - Original')
    plt.title("Shot " + repr(shot_num))
    
    diff_fig.savefig('/Users/ollie/Desktop/BNL_2020/TP/Deconvolution/Difference/shot' + repr(shot_num) +'_Difference.pdf', format='pdf', dpi=300)
    plt.close()
    
    peak_val_fig = plt.figure()
    plt.plot(peak_value)
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Peak Counts Value')
    plt.title("Shot " + repr(shot_num))
    
    peak_val_fig.savefig('/Users/ollie/Desktop/BNL_2020/TP/Deconvolution/Peak_Height/shot' + repr(shot_num) +'_Peak_Height.pdf', format='pdf', dpi=300)
    plt.close()
    
    
    
    return last_guess, peak_value;



#####################################
## Call function for all TP shots and do analysis ##


for p in np.arange(0,np.shape(filenum)[0]):
    image_path_var=filepath[p]
    shot_num=filenum[p]
    TP_Analysis_Func(image_path_var,shot_num)
    
    
    
    
    
