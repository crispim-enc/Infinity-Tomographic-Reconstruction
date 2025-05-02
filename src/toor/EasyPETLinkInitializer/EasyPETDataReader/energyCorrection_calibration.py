#Import
import numpy as np
import matplotlib.pyplot as plt
import logging

#-----------------------------------------------------
#Functions:
from dataFit_sumOfSpectrum_client import dataFit_sumOfSpectrum_client
from graphicsWithoutStatistic import graphicsWithoutStatistic
from graphicsWithStatistic import graphicsWithStatistic
from format_graph import format_graph
from dataFit_individualSpectrum_client import dataFit_individualSpectrum_client

#------------------------------------------------------

def energyCorrection_calibration(data, numberOfbins, list_energy_crystals_RCA, list_energy_crystals_RCB, list_indexes_crystals_RCA, list_indexes_crystals_RCB, view,unique_EC):

    #Parameters:
    #data = matrix of data (a1...a4..theta..phi)
    #numberOfbins = number of bins
    #list_indexs_crystals_A = events for each crystal (RCA)
    #list_indexs_crystals_B = events for each crystal (RCB)
    #list_energy_crystals_RCA = energy of events for each crystal (RCA)
    #list_energy_crystals_RCB = energy of events for each crystal (RCB)
    #view = 1 # plot information

    if (view == 1):
        #------------------------------------------------
        #FIGURES (sum of all spectrums)
        
        fig_sumOfspectrum= plt.figure(facecolor="white")
        ax_A = fig_sumOfspectrum.add_subplot(2,1,1)
        ax_B = fig_sumOfspectrum.add_subplot(2,1,2)
        list_axes_spectrum = [ax_A, ax_B]

        #------------------------------------
        #LEGENDS (information to plot)
        #legendR = [] #resolution of system and their crystals
        legend_sumOfSpectrum = []
        legend_RA = [] #resolution of system (RCA)
        legend_RB = [] #resolution of system (RCB)


        legend1 = []
        legend = []
        
       
        #---------------------------------------------------


    #SUM OF ALL SPECTRUMS = NOT CORRECTED
    
    peaks_RC = np.zeros((2,1)) #peak of each rc
    energy_matrix = np.zeros((len(data),2))
    energy_matrix[:,0] = data[:,0] + data[:,1]
    energy_matrix[:,1] = data[:,2] + data[:,3]
    
    for i in range (0,2):

        legendR = []
  
        #two RC (i = 0 (RCA) , i = 1 (RCB))

        try:
            logging.debug('Try 1 - Cadeia (0 - A e 1 - B): '+str(i))
            logging.debug('START - datafit_SumOfSpectrum')
            [xPolyfit, yPolyfit, xList, yList,
             matrixGaussFitData, lineOFfwhm, matrixOfParameters] = dataFit_sumOfSpectrum_client(energy_matrix[:,i], numberOfbins)

            logging.debug('END - datafit_SumOfSpectrum')
            peaks_RC[i,:] = matrixOfParameters[3,:]
          
            if (view == 1):
                resolution = graphicsWithStatistic (energy_matrix[:,i], numberOfbins, list_axes_spectrum[i], i,legend_sumOfSpectrum, legendR,
                                       1, xPolyfit, yPolyfit, xList, yList, matrixGaussFitData, lineOFfwhm,
                                       matrixOfParameters, 0, 2, fig_sumOfspectrum, 0)
                if (i == 0):
                    legend_RA.append(resolution)
                else:
                    legend_RB.append(resolution)

            #print(legend_sumOfSpectrum)

        except ValueError as e:
            logging.debug(e)
            [xPolyfit, yPolyfit, xList, yList] = dataFit_sumOfSpectrum_client(energy_matrix[:,i], numberOfbins)

            if (view == 1):
                #plot spectrum
                graphicsWithoutStatistic (energy_matrix[:,i], numberOfbins, list_axes_spectrum[i], i,legend, 1, 0)
                #print(legend)


    #.................................................

    if (np.sum(peaks_RC) == 0) or peaks_RC[0, :] == 0 or peaks_RC[1, :] == 0:
        #(STOP - NO RUN MORE) 
        return

    if (view == 1):
        #-------------------------------------------------
        
        #FIGURE
        fig_1 = plt.figure(facecolor="white")
        fig_2 = plt.figure(facecolor="white")
        fig_3 = plt.figure(facecolor="white")
        fig_4 = plt.figure(facecolor="white")

        fig_list = [fig_1, fig_2, fig_3, fig_4]
        

    start_point = 0
    end_point = len(unique_EC)
    
    peak_before = np.zeros((4,1))
    EA_Corrected = energy_matrix[:,0]
    EB_Corrected = energy_matrix[:,1]
    correctionFactor_Matrix = np.ones(((len(unique_EC),2)))
    list_crystals =[]

    #----------------------------------------
    #cycle (4 figures)
    #for t in range(0,4):

    n = 1
    marker = 0


    for j in range (start_point,end_point):
        logging.debug('Crystal number'+str(j))

        crystalNumber = j+1
        exist = unique_EC - crystalNumber
        find = np.where(exist == 0)

        if len(find[0] ==1):

            crystalID = find[0][0]

            #print('crystal ' +str(j))
            #cycle (16 crystals)

            # if (view == 1):
            #     #.........................................
            #     #PLOT
            #     ax_A =  fig_list[t].add_subplot(2,4,n)
            #     n+=1
            #     ax_B =  fig_list[t].add_subplot(2,4,n, sharey = ax_A)
            #     n+=1
            #     plt.setp(ax_B.get_yticklabels(), visible=False)
            #
            #     format_graph(ax_A)
            #     format_graph(ax_B)
            #     list_axes = [ax_A, ax_B]
            #
            #
            #     fig_list[t].text(0.5, 0.04, 'Energy (Kev)', ha='center')
            #     fig_list[t].text(0.04, 0.5,'Frequency', va='center', rotation = 'vertical')

                #.................................................

            for k in range(0,2):
                #cycle (2 RC)
                logging.debug('Cadeia (0 - A e 1 - B): ' + str(k))
                #print('RC' + str(k))
                legendRI = []
                #list of indexes
                if (k == 0):
                    list_energy =  list_energy_crystals_RCA[crystalID]
                    list_indexs_crystals = list_indexes_crystals_RCA[crystalID].astype(np.int64)

                else:
                    list_energy = list_energy_crystals_RCB[crystalID]
                    list_indexs_crystals = list_indexes_crystals_RCB[crystalID].astype(np.int64)

                try:
                    #dataFit_individualSpectrum(Energy, numberofbins, list_indexs_crystals)
                    logging.debug('START - dataFit_individualSpectrum_client')
                    [xPolyfit, yPolyfit, xList, yList, matrixGaussFitData, lineOFfwhm, matrixOfParameters,
                     E, xPolyfit_Corrected, yPolyfit_Corrected, xList_Corrected, yList_Corrected,
                     matrixGaussFitData_Corrected, lineOFfwhm_Corrected, matrixOfParameters_Corrected,E_Corrected, correction_factor] = dataFit_individualSpectrum_client(
                         energy_matrix[:,k], numberOfbins, list_energy, peaks_RC[k,:], crystalID,peak_before[k:k+2,:])


                    marker+= 1



                    # if (view == 1):
                    #     resolution = graphicsWithStatistic (E, numberOfbins, list_axes[k], k ,legend1, legendRI, 0, xPolyfit, yPolyfit, xList,
                    #                            yList, matrixGaussFitData, lineOFfwhm, matrixOfParameters,0,marker, fig_list[t],0)



                    list_crystals.append(crystalID)
                    peak_before = np.zeros((4,1))
                    peak_before[k,:] = matrixOfParameters[3,:]
                    peak_before[k+1,:] = matrixOfParameters_Corrected[3,:]

                    if (k == 0):

                        EA_Corrected[list_indexs_crystals] =  E_Corrected


                    else:
                        EB_Corrected[list_indexs_crystals] =  E_Corrected

                    correctionFactor_Matrix[j,k] = correction_factor
                    logging.debug('END - dataFit_individualSpectrum_client')

                    if (view == 1):
                        resolution = graphicsWithStatistic (E_Corrected, numberOfbins, list_axes[k], k,legend1, legendRI, 0, xPolyfit_Corrected,
                                               yPolyfit_Corrected, xList_Corrected, yList_Corrected, matrixGaussFitData_Corrected,
                                               lineOFfwhm_Corrected, matrixOfParameters_Corrected, 0, marker, fig_list,1)

                except TypeError as e:

                    logging.debug('EXCEPT - dataFit_individualSpectrum_client')
                    logging.debug(e)
                    dataFit_individualSpectrum_client(energy_matrix[:,k], numberOfbins, list_indexs_crystals[crystalID], peaks_RC[k,:], crystalID, peak_before[k:k+2,:])



                    if(view == 1):
                        graphicsWithoutStatistic(energy_matrix[:,k][list_indexs_crystals], numberOfbins, list_axes[k], k ,legend, 0,0)



    #start_point = end_point
    #end_point = start_point + 4
    #print(start_point)
    #print(end_point)


    #--------------------------------------------    
    # When any crystal hasn't enough counts to fit data. (STOP - NO RUN MORE)     
    if (len(list_crystals) == 0):
        return
     
    #-------------------------------------------------
    #Sum of all spectrums = CORRECTED
    # RCA
    #print(legend_sumOfSpectrum)
    try:
        #print('try')
        logging.debug('TRY - dataFit_sumOfSpectrum_client - Espectro total corrigido cadeia A')
        [xPolyfit, yPolyfit, xList, yList,
             matrixGaussFitData, lineOFfwhm, matrixOfParameters] = dataFit_sumOfSpectrum_client(EA_Corrected, numberOfbins)



        if(view == 1):
            graphicsWithStatistic (EA_Corrected, numberOfbins,list_axes_spectrum[0], 0,legend_sumOfSpectrum, legend_RA, 1, xPolyfit, yPolyfit,
                                    xList, yList, matrixGaussFitData, lineOFfwhm, matrixOfParameters, 0, 2, fig_sumOfspectrum, 1)

    except ValueError as e:
        logging.debug('Except - dataFit_sumOfSpectrum_client - Espectro total corrigido')
        logging.debug(e)
        [xPolyfit, yPolyfit, xList, yList] = dataFit_sumOfSpectrum_client(EA_Corrected, numberOfbins)
        if(view == 1):
            #plot spectrum
            graphicsWithoutStatistic (EA_Corrected, numberOfbins, list_axes_spectrum[0], 0,legend, 1, 1)


    
    # RCB
    try:
        logging.debug('TRY - dataFit_sumOfSpectrum_client - Espectro total corrigido cadeia A')
        [xPolyfit, yPolyfit, xList, yList, matrixGaussFitData, lineOFfwhm, matrixOfParameters] = dataFit_sumOfSpectrum_client(EB_Corrected, numberOfbins)

        if(view == 1):
            graphicsWithStatistic (EB_Corrected, numberOfbins,list_axes_spectrum[1], 1,legend_sumOfSpectrum, legend_RB, 1, xPolyfit, yPolyfit,
                                    xList, yList, matrixGaussFitData, lineOFfwhm, matrixOfParameters, 0, 2, fig_sumOfspectrum, 1)

    except ValueError:
        logging.debug('Except - dataFit_sumOfSpectrum_client - Espectro total corrigido')
        logging.debug(e)
        [xPolyfit, yPolyfit, xList, yList] = dataFit_sumOfSpectrum_client(EA_Corrected, numberOfbins)

        if(view == 1):
            #plot spectrum
            graphicsWithoutStatistic (EB_Corrected, numberOfbins, list_axes_spectrum[1], 1,legend, 1, 1)

    
    #----------------------------

    EA_Corrected = (EA_Corrected*511)/(peaks_RC[0][0])
    EB_Corrected = (EB_Corrected*511)/(peaks_RC[1][0])
    return EA_Corrected, EB_Corrected, correctionFactor_Matrix, peaks_RC