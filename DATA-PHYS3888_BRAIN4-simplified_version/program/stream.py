from serial.tools import list_ports
from scipy.io.wavfile import write
from classification import *
import pyautogui
import pygame
import serial
import time
import matplotlib.pyplot as plt
import pickle
import numpy as np
import warnings
import keyboard
from datetime import datetime
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names, but")

last_time_called = datetime.now()

def stream():
    """
    This method manages a continuous data stream from an E-reader via a serial port.
    It processes the incoming raw data, applies FFT filtering, uses machine learning models for classification, and provides control responses.
    It also records and visualizes the data.
    """

    classifier_filenames = [
    "Generated_Files/Random_Forest_classifier.pkl",
    "Generated_Files/SVM_classifier.pkl",
    "Generated_Files/Logistic_Regression_classifier.pkl",
    "Generated_Files/Decision_Tree_classifier.pkl",
    "Generated_Files/Gradient_Boosting_classifier.pkl",] # path of classifiers


    program_status = True # the program pauses when it is False, otherwise activated

    baudrate = 230400

    #cport = "/dev/cu.usbserial-DM7WHRHQ"
    cport = "/dev/cu.usbserial-DM02IZ0A"  # set the correct port before you run it

    ser = serial.Serial(port=cport, baudrate=baudrate)

    input_buffer_size = 10000 # keep betweein 2000-20000
    time_acquire = input_buffer_size/20000.0    # length of time that data is acquired for
    ser.timeout = input_buffer_size/20000.0  # set read timeout, 20000 is one second

    window_time = 1; # last window_time will be grabbed by classifier 
    n_window_loops = window_time/time_acquire

    tick1 = time.time()
    print("\n*********************")
    print("Starting Data Stream")
    print("*********************\n")
    classifiers = load_all_classifiers(classifier_filenames) # load all the classifiers
    classifier = classifiers["Generated Files/Random Forest classifier"] # choose the best classifier to use
    total_loop = 0 # +1 every 0.5 seconds
    # ls_event = [] # helps to check the classification

    # load the sound (indicates the status of the program)
    open = '../sound/open.wav'
    pause = '../sound/pause.ogg'

    # Start the program
    while True:
        try:

            print("\nCollecting data, loop iteration %i" %(total_loop))
            print("Eye Reader Active = %s" %(program_status))

        # Was used to test live latency
            # keyboard.on_press_key('1',left_arrow_callback)
            # keyboard.on_press_key('3',right_arrow_callback)
            # keyboard.on_press_key('2', up_arrow_callback)
            data = read_arduino(ser,input_buffer_size)
            data_temp = process_data(data)

        # if spiker box haven't collected any data, back to the beginning of while loop
            if len(data_temp) == 0:
                print("Dataset Empty, restarting loop...")
                total_loop = 0
                continue

            # Normalize the signals
            data_temp = data_temp - 512

            # Data Processing Methods:
            #t_temp = input_buffer_size/20000.0*np.linspace(0,1,len(data_temp))
            #sigma_gauss = 25
            #data_filtered_temp = process_gaussian_fft(t_temp,data_temp,sigma_gauss)
            data_filtered_temp = fft_clean(data_temp, 10000, [0,20])
            data_filtered_temp = np.real(data_filtered_temp)

            if total_loop == 0:
                data_raw_total = data_temp
                data_filtered_total = data_filtered_temp
                data_raw_window = data_temp
                data_filtered_window = data_filtered_temp
            else:
                data_filtered_total = np.append(data_filtered_total,data_filtered_temp)
                data_raw_total = np.append(data_raw_total,data_temp)

            t_filtered_total = (total_loop)*time_acquire*np.linspace(0,1,len(data_filtered_total))
            t_raw_total = (total_loop)*time_acquire*np.linspace(0,1,len(data_raw_total))

            data_last_window_filt, t_last_window_filt = get_last_window(t_filtered_total, data_filtered_total, window_time)
            data_last_window_raw, t_last_window_raw = get_last_window(t_raw_total, data_raw_total, window_time)

            #### CLASSIFICATION ####

            # predict_class is the current predicted label in the last window
            predcit_class = predict_label(data_last_window_filt, classifier)
            print("\n Movement detected: %s" % (predcit_class))
            ls_event.append(predcit_class)
            if predcit_class == "N":
                # filter out some biases when classifying
                if len(ls_event) > 3:
                    # print(ls_event)
                    # if double blink satisfied, switch the program status and play corresponding music.
                    if ls_event.count("B") >= 6:
                        # write_time(1)
                        play_music(pause) if program_status else play_music(open)
                        program_status = not program_status
                    # a metric to increase live accuracy, we choose to select the most frequent one occurs in the period of time.
                    # Also, we delete the head and the tail because they are less reliable
                    ls_event = deleteHeadAndTail(ls_event)
                    unique_elements, counts = np.unique(ls_event, return_counts=True)
                    most_frequent_element = unique_elements[np.argmax(counts)]
                    if program_status and most_frequent_element == "R":
                        # write_time(1)
                        goRight()
                    elif program_status and most_frequent_element == "L":
                        # write_time(1)
                        goLeft()
                ls_event = [] # make a new list for next movement
            else:
                ls_event.append(predcit_class) # if an event detected, keep adding possible movement into this list
            total_loop = total_loop + 1
        except KeyboardInterrupt:
            break

    print("\n*********************")
    print("Data Stream Ended")
    print("*********************\n")
    tick2 = time.time()
    time_taken = tick2 - tick1
    print("Time taken: ",time_taken)

    # Plot data: 
    fig1, ax1 = plt.subplots()
    ax1.clear()
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude (nd)")
    ax1.set_title("Last %i second of stream" %(window_time))
    ax1.plot(t_last_window_raw,data_last_window_raw, label = "Raw data")
    ax1.plot(t_last_window_filt,data_last_window_filt, label = "FFT filtered data")
    ax1.legend()
    ax1.set_ylim([-500, 500])

    fig2, ax2 = plt.subplots()
    ax2.clear()
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude (nd)")
    ax2.set_title("Stream")
    ax2.plot(t_raw_total, data_raw_total, label = "Raw data")
    ax2.plot(t_filtered_total, data_filtered_total, label = "FFT filtered data")
    ax2.legend()
    ax2.set_ylim([-500, 500])
    plt.show()

    # Save data set 
    sample_rate = 10000
    file_name = "Streamed_Data/stream"

    # Save as wav
    datasave = np.column_stack((t_filtered_total,data_filtered_total))
    write("%s.wav" %file_name, sample_rate, datasave.astype(np.int16))

    # Save image
    fig2.savefig("%s.png" %(file_name))
    
    # Flush Ports
    if ser.read():
        ser.flushInput()
        ser.flushOutput()
        ser.close()

def left_arrow_callback(e):
    global last_time_called
    now = datetime.now()
    time_diff = now - last_time_called
    if time_diff.seconds < 1:   # If less than 1 second has passed, return
        return
    last_time_called = now
    with open('test_trail.txt', 'a') as f:
        f.write('Left arrow key pressed at: ' + str(datetime.now()) + ',')
def right_arrow_callback(e):
    global last_time_called
    now = datetime.now()
    time_diff = now - last_time_called
    if time_diff.seconds < 1:   # If less than 1 second has passed, return
        return
    last_time_called = now
    with open('test_trail.txt', 'a') as f:
        f.write('Right arrow key pressed at: ' + str(datetime.now()) + ',')
def write_time(e):
    with open('test_trail.txt', 'a') as f:
        f.write(str(datetime.now()) + '\n')

def up_arrow_callback(e):
    global last_time_called
    now = datetime.now()
    time_diff = now - last_time_called
    if time_diff.seconds < 1:   # If less than 1 second has passed, return
        return
    last_time_called = now
    with open('test_trail.txt', 'a') as f:
        f.write('Blink arrow key pressed at: ' + str(datetime.now()) + ',')

def read_arduino(ser,input_buffer_size):
    data = ser.read(input_buffer_size)
    out =[(int(data[i])) for i in range(0,len(data))]
    return out

def process_data(data):
    data_in = np.array(data)
    result = []
    i = 1
    while i < len(data_in)-1:
        if data_in[i] > 127:
            # Found beginning of frame
            # Extract one sample from 2 bytes
            intout = (np.bitwise_and(data_in[i],127))*128
            i = i + 1
            intout = intout + data_in[i]
            result = np.append(result,intout)
        i=i+1
    return result

def filter_out(f_min, f_max, freq_ft, data_ft): # this method will reduce amplitude of data_ft to 0 for all f outside of range(fmin, fmax)
    done_min = False 
    done_max = False
    index_min = 0
    index_max = 0
    for i in range(0,len(freq_ft)): 
        if freq_ft[i] >= f_min and done_min == False: 
            index_min =i
            done_min = True
        if freq_ft[i] >= f_max and done_max == False: 
            index_max = i 
            done_max = True
    data_ft_clean = data_ft
    data_ft_clean[index_max:]= 0
    data_ft_clean[0:index_min] = 0

    return data_ft_clean 

def fft_clean(data, sample_rate,frequency_range): 
    f_min = frequency_range[0]
    f_max = frequency_range[1]

    data_ft = np.fft.fft(data)
    freq = np.linspace(0, sample_rate,len(data_ft))

    data_ft_clean = filter_out(f_min,f_max,freq, data_ft)
    data_clean = np.fft.ifft(data_ft_clean)
    
    return data_clean

def process_gaussian_fft(t,data_t,sigma_gauss):
    nfft = len(data_t) # number of points

    dt = t[1]-t[0]  # time interval
    maxf = 1/dt     # maximum frequency
    df = 1/np.max(t)   # frequency interval
    f_fft = np.arange(-maxf/2,maxf/2+df,df)          # define frequency domain

    ## DO FFT
    data_f = np.fft.fftshift(np.fft.fft(data_t)) # FFT of data

    ## GAUSSIAN FILTER
    #    sigma_gauss = 25  # width of gaussian - defined in the function
    gauss_filter = np.exp(-(f_fft)**2/sigma_gauss**2)   # gaussian filter used
    print("gauss filt length = %i" %(len(gauss_filter)))
    print(len(gauss_filter))
    data_f_filtered= data_f*gauss_filter    # gaussian filter spectrum in frquency domain
    data_t_filtered = np.fft.ifft(np.fft.ifftshift(data_f_filtered))    # bring filtered signal in time domain
    return data_t_filtered

def find_ports():
    ports = list_ports.comports()
    print("\n")
    for port in ports:
        print("Port option:  %s "%(port))
    print("\n")

def get_last_window(time_array, data_array, window): 
    start_index= 0
    data_last_window = data_array
    time_last_window = time_array
    for i in range(len(time_array)-1, 0, -1): 
        if (time_array[-1] - time_array[i]) > window: 
            start_index = i
            break
    data_last_window = data_last_window[start_index:]
    time_last_window = time_last_window[start_index:]
    return data_last_window, time_last_window

def goLeft():
    pyautogui.press('left')

def goRight():
    pyautogui.press('right')

def play_music(file_path):
    """
    This function defines a way to play the music to indicates the status of the program
    """
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
