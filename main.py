""""""""""""""""""""" MODULE IMPORT """""""""""""""""""""
### Import external modules
import os
import matplotlib
matplotlib.use('Agg') #to run the script on a distant server
import matplotlib.pyplot as plt
from scipy import interpolate
import os.path
import subprocess
from shutil import copyfile
import os
import sys

sys.path.insert(0, os.path.realpath('..'))
sys.path.insert(0, os.path.realpath('../Classes'))
sys.path.insert(0, os.path.realpath('../Functions'))
sys.path.insert(0, os.path.realpath('Classes'))
sys.path.insert(0, os.path.realpath('Functions'))

### Import internal functions
from Functions.create_hidden_variables import create_hidden_variables

### Import supplementary analysis
from SupplementaryAnalysis.circadian_speed import circadian_speed
from SupplementaryAnalysis.coupling_proof import test_coupling
from SupplementaryAnalysis.phase_delay import phase_delay
from SupplementaryAnalysis.test_correlations import test_correlations
from SupplementaryAnalysis.test_phase_diffusion import test_sigma_theta

""""""""""""""""""""" FUNCTION """""""""""""""""""""

def main(cell = "NIH3T3", temperature = None, nb_trace = 20, nb_iter = 3,
         size_block = 100):

    #access script directory
    os.chdir('Scripts')

    """"" START WITH MAIN PIPELINE """""
    wrap_initial_parameters = False#True
    optimize_parameters_non_dividing = False#True
    compute_bias = False#True
    CV_smoothing = False#True
    optimize_parameters_dividing = False#True
    validate_inference_non_dividing = False#True
    validate_estimation_non_dividing = False#True
    validate_inference_dividing = False#True
    compute_final_fits_and_attractor = True
    if cell=="NIH3T3" and temperature is None and nb_trace>1000:
        compute_final_fits_and_attractor_by_period = True
    else:
        compute_final_fits_and_attractor_by_period = False
    study_deterministic_system = True
    study_stochastic_system = True

    if wrap_initial_parameters:
        """ GUESS OR ESTIMATE PARAMETERS ON NON-DIVIDING TRACES """
        #remove potential previous results
        path = '../Parameters/Real/init_parameters_nodiv_'+str(temperature)\
                +"_" + cell+".p"
        if os.path.isfile(path) :
            os.remove(path)
        #run script
        print("1_wrap_initial_parameters launched")
        os.system("python 1_wrap_initial_parameters.py " + cell + " "
                + str(temperature)
                + ' > ../TextOutputs/1_wrap_initial_parameters_'
                +cell+'_'+str(temperature)+'.txt')
        #check if final result has been created
        if os.path.isfile(path) :
            print("Initial set of parameters successfully created")
        else:
            print("BUG, initial set of parameters not created")

    if optimize_parameters_non_dividing:
        """ OPTIMIZE WAVEFORM ON NON-DIVIDING TRACES """
        #remove potential previous results
        path = "../Parameters/Real/opt_parameters_nodiv_"+str(temperature)+"_"\
                + cell+".p"
        if os.path.isfile(path) :
            os.remove(path)
        #run script
        print("2_optimize_parameters_non_dividing launched")
        os.system("python 2_optimize_parameters_non_dividing.py "
                  + str(nb_iter) + " "+ str(nb_trace) + " "+  str(size_block)
                  + " "+cell + " "+ str(temperature)
                  + ' > ../TextOutputs/2_optimize_parameters_non_dividing_'+cell
                  +'_'+str(temperature)+'.txt')
        #check if final result has been created
        if os.path.isfile(path) :
            print("Optimized set of parameters on non-dividing traces \
                   successfully created")
        else:
            print("BUG, optimized set of parameters on non-dividing traces \
                   not created")

    #case no optimization is done for non-dividing parameters, then take
    #directly estimated parameters
    else:
        path = "../Parameters/Real/opt_parameters_nodiv_"+str(temperature)\
                +"_"+cell+".p"
        #if not os.path.isfile(path):
        copyfile('../Parameters/Real/init_parameters_nodiv_'+str(temperature)\
                +"_"+cell+".p", path)


    if compute_bias:
        """ COMPUTE WAVEFORM BIAS """
        #remove potential previous results
        path = "../Parameters/Misc/F_no_coupling_"+str(temperature)+"_"\
               +cell+'.p'
        if os.path.isfile(path) :
            os.remove(path)
        #run script
        print("7_bias launched")
        os.system("python 7_validate_inference_dividing.py "
                  + str(nb_iter) + " "+ str(nb_trace) + " "+  str(size_block)
                  + " "+cell + " "+ str(temperature) + " True False"
                  + ' > ../TextOutputs/7_bias_'+cell+'_'
                  +str(temperature)+'.txt')
        #check if final result has been created
        if os.path.isfile(path) :
            print("Bias successfully computed")
        else:
            print("BUG, bias not computed")

    if CV_smoothing:
        """ COMPUTE CROSS-VALIDATION FOR SMOOTHING """
        #remove potential previous results
        path = 'Results/Smoothing/CV_'+str(temperature)+"_"+cell+'.pdf'
        if os.path.isfile(path) :
            os.remove(path)
        #run script
        print("3_CV_smoothing launched")
        os.system("python 3_CV_smoothing.py " + str(nb_iter) + " "\
                  + str(nb_trace) + " "+  str(size_block) + " "+cell + " "\
                  + str(temperature)+ ' > ../TextOutputs/3_CV_smoothing_'\
                  +cell+'_'+str(temperature)+'.txt')
        #check if final result has been created
        if os.path.isfile(path) :
            print("Smoothing parameter successfully computed")
        else:
            print("BUG, smoothing parameter not computed")

    if optimize_parameters_dividing:
        """ OPTIMIZE COUPLING ON DIVIDING TRACES """
        #remove potential previous results
        path = "../Parameters/Real/opt_parameters_div_"+str(temperature)+"_"\
                +cell+".p"
        if os.path.isfile(path) :
            os.remove(path)
        #run script
        print("4_optimize_parameters_dividing launched")
        os.system("python 4_optimize_parameters_dividing.py "
                  + str(nb_iter) + " "+ str(nb_trace) + " "+  str(size_block)
                  + " "+cell + " "+ str(temperature)
                  + ' > ../TextOutputs/4_optimize_parameters_dividing_'+cell+'_'
                  +str(temperature)+'.txt')
         #check if final result has been created
        if os.path.isfile(path) :
            print("Optimized set of parameters on dividing traces \
                   successfully created")
        else:
            print("BUG, optimized set of parameters on dividing traces \
                   not created")

    if validate_inference_non_dividing:
        """ VALIDATE INFERENCE WAVEFORM ON NON-DIVIDING TRACES """
        #remove potential previous results
        path = "../Parameters/Silico/opt_parameters_nodiv_"+str(temperature)\
                +"_"+cell+'.p'
        if os.path.isfile(path) :
            os.remove(path)
        #run script
        print("5_validate_inference_non_dividing launched")
        os.system("python 5_validate_inference_non_dividing.py "
                  + str(nb_iter) + " "+ str(nb_trace) + " "+  str(size_block)
                  + " "+cell + " "+ str(temperature)
                  + ' > ../TextOutputs/5_validate_inference_non_dividing_'+cell
                  +'_'+str(temperature)+'.txt')
         #check if final result has been created
        if os.path.isfile(path) :
            print("Validation of the parameters optimization on non-dividing \
                  traces succesfuly done")
        else:
            print("BUG, validation of the parameters optimization on \
                  non-dividing traces not done")

    if validate_estimation_non_dividing:
        """ VALIDATE PARAMETERS ESTIMATION ON NON-DIVIDING TRACES """
        #remove potential previous results
        path =  "../Parameters/Silico/est_parameters_nodiv_" + str(temperature)\
                + "_" + cell+'.p'
        if os.path.isfile(path) :
            os.remove(path)
        #run script
        print("6_estimation_non_dividing launched")
        os.system("python 6_validate_estimation_non_dividing.py "
                  + str(nb_trace) + " "+cell + " "+ str(temperature)
                  + ' > ../TextOutputs/6_validate_estimation_non_dividing_'+cell
                  +'_'+str(temperature)+'.txt')
         #check if final result has been created
        if os.path.isfile(path) :
            print("Validation of the parameters estimation on non-dividing \
                  traces succesfuly done")
        else:
            print("BUG, validation of the parameters estimation on non-dividing\
                  traces not done")

    if validate_inference_dividing:
        """ VALIDATE COUPLING INFERENCE ON DIVIDING TRACES """
        #remove potential previous results
        path = "../Parameters/Silico/opt_parameters_div_"+str(temperature)+"_" \
                + cell +'_'+'False'+'.p'
        if os.path.isfile(path) :
            os.remove(path)
        #run script
        print("7_validate_inference_dividing launched")
        os.system("python 7_validate_inference_dividing.py "
                  + str(nb_iter) + " "+ str(nb_trace) + " "+  str(size_block)
                  + " "+cell + " "+ str(temperature) + " False True"
                  + ' > ../TextOutputs/7_validate_inference_dividing_'+cell+'_'
                  +str(temperature)+'.txt')
        os.system("python 7_validate_inference_dividing.py "
                  + str(nb_iter) + " "+ str(nb_trace) + " "+  str(size_block)
                  + " "+cell + " "+ str(temperature) + " False False"
                  + ' > ../TextOutputs/7_validate_inference_dividing_'+cell
                  +'_'+str(temperature)+'.txt')

          #check if final result has been created
        if os.path.isfile(path) :
            print("Validation of the coupling inference on dividing traces \
                   succesfuly done")
        else:
            print("BUG, validation of the coupling inference on non-dividing \
                  traces not done")

    if compute_final_fits_and_attractor:
        """ COMPUTE FINAL FITS AND ATTRACTOR """
        #remove potential previous results
        path = "../Results/PhaseSpace/PhaseSpaceDensitySuperimposed_"\
                +str(temperature)+"_"+cell+'_None.pdf'
        if os.path.isfile(path) :
            os.remove(path)        #run script
        print("8_compute_final_fits_and_attractor launched")
        os.system("python 8_compute_final_fits_and_attractor.py "
                  + str(nb_trace) + " "+  cell + " "+ str(temperature)
                  +" None"
                  + ' > ../TextOutputs/8_compute_final_fits_and_attractor_'
                  +cell+'_'+str(temperature)+'.txt')
          #check if final result has been created
        if os.path.isfile(path) :
            print("Final fits and attractor computed")
        else:
            print("BUG, final fits and attractor not computed")

    if compute_final_fits_and_attractor_by_period:
        """ COMPUTE FINAL FITS AND ATTRACTOR BY PERIOD """
        l_period = [12,16,20,24,28,32,36]
        for period in l_period:
            #remove potential previous results
            path = "../Results/PhaseSpace/PhaseSpaceDensitySuperimposed_"\
                    +str(temperature)+"_"+cell+"_"+str(period)+'.pdf'
            if os.path.isfile(path) :
                os.remove(path)
            #run script
            print("8_compute_final_fits_and_attractor by period launched, \
                   period ", period)
            os.system("python 8_compute_final_fits_and_attractor.py "
                      + str(nb_trace) + " "+  cell + " "+ str(temperature)
                      + " "+ str(period)
                      + ' > ../TextOutputs/8_compute_final_fits_and_attractor_'
                      +cell+'_'+str(temperature)+"_"+str(period)+'.txt')
              #check if final result has been created
            if os.path.isfile(path) :
                print("Final fits and attractor computed for period ", period)
            else:
                print("BUG, final fits and attractor not computed for period ",
                      period)

    if study_deterministic_system:
        """ COMPUTE RESULTS FROM DETERMINISTIC SYSTEM """
        #remove potential previous results
        path = '../Results/DetSilico/devil_'+cell+'_'+str(temperature)\
               +'.pdf'
        if os.path.isfile(path) :
            os.remove(path)
        #run script
        print("9_study_deterministic_system launched")
        os.system("python 9_study_deterministic_system.py "   +cell
                  + " "+ str(temperature)
                  +  ' > ../TextOutputs/9_study_deterministic_system_'+cell+'_'
                  +str(temperature)+'.txt')
          #check if final result has been created
        if os.path.isfile(path) :
            print("Results from deterministc system succesfully computed")
        else:
            print("BUG, results from deterministc system not computed")

    if study_stochastic_system:
        """ COMPUTE RESULTS FROM STOCHASTIC SYSTEM """
        #remove potential previous results
        path = '../Results/StochasticSilico/CountGenerated_'+cell+'_'\
                +str(temperature)+'.pdf'
        if os.path.isfile(path) :
            os.remove(path)
        #run script
        print("10_study_stochastic_system launched")
        os.system("python 10_study_stochastic_system.py "
                  + str(nb_trace) +  " "+cell + " "+ str(temperature)
                  + ' > ../TextOutputs/10_study_stochastic_system_'+cell+'_'
                  +str(temperature)+'.txt')
          #check if final result has been created
        if os.path.isfile(path) :
            print("Results from stochastic system succesfully computed")
        else:
            print("BUG, results from stochastic system not computed")

main(cell = "NIH3T3", temperature = 37, nb_trace = 200, nb_iter = 5,
     size_block = 100)
print("main finished")
