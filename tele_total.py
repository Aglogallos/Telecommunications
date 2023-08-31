import numpy as np                      #for arrays and matrices
import matplotlib.pyplot as plt         #for ploting
import scipy                            #for math calculations
from mpl_toolkits.mplot3d import Axes3D #for 3D ploting
import math                             #for math constants
from matplotlib import collections as matcoll
from scipy import signal                #for signal analysis
from scipy.fftpack import fft           #for fourier spectrum

#Aglogallos Anastasios 
#031 18641

########### Άσκηση 1  #########

##PARAMETERS

Tm = 1 / 3000 #sec
A = 4 #V
AM = 2 #6 + 4 + 1 = 11 = 1+1
Fm = AM #kHz
N_periods = 4 #periods displayed

#INITIAL SIGNAL
t = np.linspace(0, N_periods*Tm, 8000+1)
#y = A*np.cos(2*math.pi*Fm*t)*np.cos(2*math.pi*(AM+2)*Fm*t)
y = A * np.abs(signal.sawtooth(2 * np.pi * Fm * time)) #triangle waveform
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0)) #for automatic x-scale (10^-3)
plt.plot(t, y, '-', marker="", markersize=4) #for markers
plt.show()

#Α ΕΡΏΤΗΜΑ
#version 1
def sign_sampling(coeff):
    t_freq = np.linspace(0, N_periods*Tm, N_periods*coeff+1)
    y_freq = A * np.abs(signal.sawtooth(2 * np.pi * Fm * t_freq))
    plt.vlines(t_freq, [0], y_freq, linewidth=0.3)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0)) #for automatic x-scale (10^-3)
    plt.plot(t_freq, y_freq, '.', marker=".", markersize=4)
    # plt.plot(t, y, '-', t_freq, y_freq, '.', N_periods*Tm, y[0])
    plt.show()

#version 2
def sign_sampling_2(coeff):
    t_freq = np.linspace(0, N_periods*Tm, N_periods*coeff)
    y_freq = A * np.abs(signal.sawtooth(2 * np.pi * Fm * t_freq))
    plt.plot(t, y, '-', t_freq, y_freq, '.', N_periods*Tm, y[0])
    plt.show()

#Ι) -> fs1=20fm
##### version 1 #####
# t_20 = np.linspace(0, 4/3000, 80, endpoint=False)
# y_20 = signal.resample(y, 80)
# plt.vlines(t_20, [0], y_20, linewidth=0.3)
# plt.plot(t_20, y_20, '.', marker=".", markersize=4)
# plt.show()

##### version 2 #####
# t_20 = np.linspace(0,4/3000,80)
# y_20 = A*np.cos(2*math.pi*3000*t_20)*np.cos(2*math.pi*(AM+2)*3000*t_20)
# plt.vlines(t_20, [0], y_20, linewidth=0.3)
# plt.plot(t_20, y_20, '.', marker=".", markersize=4)
# plt.show()

##### version 3 #####
sign_sampling(30)

##### version 4 #####
# t = np.arange(0, 4/3000, 1/(100*3000))
# y = A*np.cos(2*math.pi*3000*t)*np.cos(2*math.pi*(AM+2)*3000*t)
# plt.stem(t, y, use_line_collection=True)
# plt.show()


#ΙΙ) -> fs2=100fm 
##### version 1 #####
# t_100 = np.linspace(0, 4/3000, 400, endpoint=False)
# y_100 = signal.resample(y, 400)
# plt.vlines(t_100, [0], y_100, linewidth=0.3)
# plt.plot(t_100, y_100, '.', marker=".", markersize=4)
# plt.show()

##### version 2 #####
# t_100 = np.linspace(0,4/3000,400)
# y_100 = A*np.cos(2*math.pi*3000*t_100)*np.cos(2*math.pi*(AM+2)*3000*t_100)
# plt.vlines(t_100, [0], y, linewidth=0.3)
# plt.plot(t_100, y_100, '.', marker=".", markersize=4)
# plt.show()

##### version 3 #####
sign_sampling(50)

#III)
t_20 = np.linspace(0, N_periods*Tm, 80+1)
y_20 = A*np.cos(2*math.pi*Fm*t_20)*np.cos(2*math.pi*(AM+2)*Fm*t_20)
plt.vlines(t_20, [0], y_20, linewidth=0.8, colors="b")

t_100 = np.linspace(0, N_periods/Fm, 400+1)
y_100 = A*np.cos(2*math.pi*Fm*t_100)*np.cos(2*math.pi*(AM+2)*Fm*t_100)
plt.vlines(t_100, [0], y_100, linewidth=0.3)

plt.plot(t_20, y_20, '.', t_100, y_100, '.', 4/3000, y[0])
plt.legend(['fs1', 'fs2'])
plt.show()

#B ΕΡΏΤΗΜΑ
# t_5 = np.linspace(0,4/3000,20)
# y_5 = A*np.cos(2*math.pi*3000*t_5)*np.cos(2*math.pi*(AM+2)*3000*t_5)
# plt.vlines(t_5, [0], y_5, linewidth=0.3)
# plt.plot(t_5, y_5, '.', marker=".", markersize=4)
# plt.show()
sign_sampling(4)

#fourier spectrum
signal_fft = fftpack.fft(y)
Amplitude = np.abs(signal_fft)
Power = Amplitude**2
Angle = np.angle(signal_fft)
sample_freq = fftpack.fftfreq(y.size, d=Tm/8001)
Amp_freq = np.array([Amplitude, sample_freq])

plt.plot(abs(signal_fft), "o")
plt.show()

plt.subplot(2, 1, 1)
plt.plot(t, y)


t = np.linspace(0, N_periods*Tm, 8000+1)
y = A*np.cos(2*math.pi*Fm*t)*np.cos(2*math.pi*(AM+2)*Fm*t)
yf = fft(y)
tf = np.linspace(0.0, 1.0/(2.0*Tm), 8001//2)

plt.plot(tf, 2.0/8001 * np.abs(yf[0:8001//2]))
plt.grid()
plt.show()


from scipy.fft import fft
# Number of sample points
N = 600
# sample spacing
T = 1.0 / 800.0
x = np.linspace(0.0, N*T, N)
y = A*np.cos(2*math.pi*Fm*x)*np.cos(2*math.pi*(AM+2)*Fm*x)
yf = fft(y)
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
import matplotlib.pyplot as plt
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.grid()
plt.show()

########### FINAL #######
#number of samples
N = 100*4
# sample spacing
T = 1.0 / (100.0*3000.0)
x = np.linspace(0.0, N*T, N)
y = np.cos(3000.0 * 2.0*np.pi*x)*np.cos(5.0*3000.0 * 2.0*np.pi*x)
plt.subplot(2, 1, 1)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0)) #for automatic x-scale (10^-3)
plt.plot(x, y)
yf = fft(y)
xf = fftpack.fftfreq(N, T)
xf = fftpack.fftshift(xf)
yplot = fftpack.fftshift(yf)
plt.subplot(2, 1, 2)
plt.xlim(0, 8*3000)
plt.xticks(np.arange(0, 11*3000, step=3000))
plt.plot(xf, 4.0/N * np.abs(yplot))
plt.grid()
plt.show()

########### Άσκηση 2  #########

##PARAMETERS
Fm = 3000 #kHz
Tm = 1 / Fm #sec
A = 1 #V
AM = 3 
N_periods = 4 #periods displayed

Samples_per_period = 2000 #number of samples per period
N_samples = N_periods * Samples_per_period + 1 #total number of samples (in linspace)
Timestep = 1.0 / (float(Fm * Samples_per_period)) #sample spacing

#### Α ΕΡΏΤΗΜΑ ####
#SIGNAL (20fm)
t_20 = np.linspace(0, N_periods*Tm, 4*20+1)
y_20 = A*np.cos(2*math.pi*Fm*t_20)*np.cos(2*math.pi*(AM+2)*Fm*t_20)

# plt.vlines(t_20, [0], y_20, linewidth=0.8, colors="b")
# plt.plot(t_20, y_20, '.')
# plt.show()

#######################################
bits = 5                      #bits for quantization
q_levels = 2**bits              #quantization levels
q_levels_top = q_levels/2       #quantization levels on one side               
s_max = max(abs(y_20))          #get the max value
delta = (2*s_max)/(q_levels-1)  #step size

#QUANTIZATION
quant_signal = np.copy(y_20) #np.copy() copies y_20 array without reference
y_20_new = np.copy(y_20)
for i in range(0,y_20.size):
    quant_signal[i] = int(math.floor(round(y_20[i],4)/delta)) #quantized levels (int)
    y_20_new[i] = delta*(quant_signal[i])+delta/2 #mid-riser quantization
    # print(str(y_20[i]) +' : '+ str(y_20_new[i]))
    # print(quant_signal[i])

#GRAY CODE GENERATOR
def gray_code(n_bits):
    gray_arr = list()
    gray_arr.append("0")
    gray_arr.append("1")
    i = 2
    j = 0
    while(True):
        if i>=1 << n_bits:
            break
        for j in range(i - 1, -1, -1):
            gray_arr.append(gray_arr[j])
        for j in range(i):
            gray_arr[j] = "0" + gray_arr[j]
        for j in range(i, 2 * i):
            gray_arr[j] = "1" + gray_arr[j]
        i = i << 1

    return gray_arr

#PLOT FOR QUANTIZED SIGNAL
gray_code_ex2 = gray_code(bits)
plt.vlines(t_20, [0], quant_signal, linewidth=0.8, colors="b")
plt.yticks(np.arange(-q_levels/2, q_levels/2, 1), gray_code_ex2)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0)) #for automatic x-scale (10^-3)
plt.plot(t_20, quant_signal, '.', label='fs1=20fm')
plt.title('y(t) quantized with mid riser')   
plt.xlabel('Time (sec)'); plt.ylabel('Gray Code')
plt.legend(loc='upper left')
# plt.show()
plt.figure()


#### Β ΕΡΏΤΗΜΑ ####
#Ι) -> variance (τυπική απόκλιση) for 10 samples 
error_10 = y_20[0:10]-y_20_new[0:10]
var_10 = (1/10)*sum(map(lambda x:x*x,error_10))
print('Variance for first \n10 samples : '+ str(var_10))

#ΙI) -> variance (τυπική απόκλιση) for 20 samples
error_20 = y_20[0:20]-y_20_new[0:20]
var_20 = (1/20)*sum(map(lambda x:x*x,error_20))
print('20 samples : '+ str(var_20))
        
#III) -> SNR
error_quant = (1/3)*pow(A,2)*pow(2,-2*bits) #Τυπική απόκλιση σφάλματος κβάντισης
P_mean_20 = (1/20)*sum(map(lambda x:x*x,y_20[0:20])) #Mέση ισχύς του σήματος y_20(t)
SNR_10 = P_mean_20 / var_10 #SNR for 10 samples
print('SNR for 10 samples : '+ str(SNR_10))
SNR_20 = P_mean_20 / var_20 #SNR for 20 samples
print('SNR for 20 samples : '+ str(SNR_20))
SNR_theor = P_mean_20 / error_quant #SNR for theoretical value
print('SNR for theoretical : '+ str(SNR_theor))


#### Γ ΕΡΏΤΗΜΑ ####
Bitstream = '' #bit stream of output (string)
polar_nrz = []
for i in range(0, 20):
    Bitstream += gray_code_ex2[int(quant_signal[i]+q_levels/2)] #creates string of bitstream
#Option A
samples_per_bit = 100
for i in range(0, len(Bitstream)):
    for j in range(0, samples_per_bit):
        polar_nrz.append(int(Bitstream[i])) #appends bits to array
#Option B
# polar_nrz = list(Bitstream)


t_bit_20 = np.linspace(0, 0.001*bits*20, samples_per_bit*bits*20, endpoint=False)
plt.plot(t_bit_20, Fm/1000*signal.square(2*math.pi*t_bit_20, duty=polar_nrz[0:samples_per_bit*bits*20]), label='POLAR NRZ')
plt.grid(True, which='both')
plt.axhline(y=0, color='k')
plt.title('Bit stream of quantized signal (fs1=20fm)')   
plt.xlabel('Time (sec)'); plt.ylabel('Amplitude (V)')
plt.legend(loc='upper left')
plt.show()


###### Άσκηση 3 ########


Samples_per_period = 2000 #number of samples per period
N_samples = N_periods * Samples_per_period + 1 #total number of samples (in linspace)
Timestep = 1.0 / (float(Fm * Samples_per_period)) #sample spacing

#### Α ΕΡΏΤΗΜΑ ####
A_bit = Fm/1000 #(V) Amplitude of bit stream
T_b = 0.5 #(sec) bit duration 
N_rand_bits = 46 #number of random bits generated

rand_bits = np.random.randint(2, size=(N_rand_bits)) #generate random bits [0,1]

samples_per_bit = 100
rand_bits_linspace = []
for i in range(0, len(rand_bits)):  
    for j in range(0, samples_per_bit):
        rand_bits_linspace.append(rand_bits[i])

t_rand_bits = np.linspace(0, T_b*N_rand_bits, N_rand_bits*samples_per_bit, endpoint=False)
y_rand_bits = A_bit*signal.square(2*math.pi*t_rand_bits, duty=rand_bits_linspace[0:N_rand_bits*samples_per_bit])
plt.plot(t_rand_bits, y_rand_bits, label='B-PAM')
plt.grid(True, which='both')
plt.axhline(y=0, color='k')
plt.title('B-PAM modulation of random bits')   
plt.xlabel('Time (sec)'); plt.ylabel('Amplitude (V)')
plt.legend(loc='upper left')
# plt.show()
plt.figure()

#### B ΕΡΏΤΗΜΑ ####
E_b = pow(A_bit, 2)*T_b 

x_bpam = [-math.sqrt(E_b), math.sqrt(E_b)]
y_bpam = [0, 0]
plt.scatter(x_bpam,y_bpam)
plt.ylim([-0.5, 0.5])
plt.grid(True, which='both')
plt.title('Constellation of B-PAM')   
plt.xlabel('In-Phase'); plt.ylabel('Quadrature')
# plt.show()
plt.figure()

#### Γ ΕΡΏΤΗΜΑ ####
#SNR to linear scale function
def SNR_dB_lin(snr_ratio):
    return 10**(snr_ratio / 10)

No_5 = E_b / SNR_dB_lin(5) #Conversion from dB to linear scale
No_15 = E_b / SNR_dB_lin(15) #Conversion from dB to linear scale

plt.subplots_adjust(hspace=0.5)
#Eb/No (SNR) = 5 dB
# awgn_5 = np.sqrt(No_5/2)*np.random.standard_normal(N_rand_bits*samples_per_bit)
awgn_5 = np.random.normal(0, np.sqrt(No_5), 2*N_rand_bits*samples_per_bit).view(np.complex128) #complex awgn (5dB)
# awgn_5_im = np.random.normal(0, math.sqrt(No_5), N_rand_bits*samples_per_bit) #imaginary part of awgn (5dB)
plt.subplot(3, 1, 2)
plt.plot(t_rand_bits, y_rand_bits + awgn_5.real, label='Eb/No = 5')
plt.grid(True, which='both')
plt.axhline(y=0, color='k')
plt.title('Eb/No=5')   
plt.ylabel('Amplitude (V)')
plt.legend(loc='upper left')

#Eb/No (SNR) = 15 dB
# awgn_15 = np.sqrt(No_15/2)*np.random.standard_normal(N_rand_bits*samples_per_bit)
awgn_15 = np.random.normal(0, np.sqrt(No_15), 2*N_rand_bits*samples_per_bit).view(np.complex128) #complex awgn (15dB)
# awgn_15_im = np.random.normal(0, math.sqrt(No_15), N_rand_bits*samples_per_bit) #imaginary part of awgn (15dB)
plt.subplot(3, 1, 3)
plt.plot(t_rand_bits, y_rand_bits + awgn_15.real, label='Eb/No = 15')
plt.grid(True, which='both')
plt.axhline(y=0, color='k')
plt.title('Eb/No=15')   
plt.xlabel('Time (sec)'); plt.ylabel('Amplitude (V)')
plt.legend(loc='upper left')

#Initial B-PAM signal
plt.subplot(3, 1, 1)
plt.plot(t_rand_bits, y_rand_bits, label='B-PAM')
plt.grid(True, which='both')
plt.axhline(y=0, color='k')
plt.title('B-PAM modulation of random bits')   
plt.ylabel('Amplitude (V)')
plt.legend(loc='upper left')
plt.show()
# plt.figure()

#### Δ ΕΡΏΤΗΜΑ ####
#Eb/No = 5 dB
x_bpam_awgn_5 = (y_rand_bits[::1] + awgn_5.real[::1]) * math.sqrt(T_b)
y_bpam_awgn_5 = (awgn_5.imag[::1]) * math.sqrt(T_b)
# y_bpam_awgn_5 = np.zeros(N_rand_bits*samples_per_bit // 50)

plt.subplot(2, 1, 1)
plt.scatter(x_bpam_awgn_5 ,y_bpam_awgn_5, s=2.5, c='b', label='Eb/No = 5')
plt.ylim([-0.5, 0.5])
plt.grid(True, which='both')
plt.title('Constellation of B-PAM')   
plt.legend(loc='upper left')
plt.ylabel('Quadrature')

#Eb/No = 15 dB
x_bpam_awgn_15 = (y_rand_bits[::1] + awgn_15.real[::1]) * math.sqrt(T_b)
y_bpam_awgn_15 = (awgn_15.imag[::1]) * math.sqrt(T_b)
# y_bpam_awgn_15 = np.zeros(N_rand_bits*samples_per_bit // 50)

plt.subplot(2, 1, 2)
plt.scatter(x_bpam_awgn_15 ,y_bpam_awgn_15, s=2, c='b', label='Eb/No = 15')
plt.ylim([-0.5, 0.5])
plt.grid(True, which='both') 
plt.xlabel('In-Phase'); plt.ylabel('Quadrature')
plt.legend(loc='upper left')
plt.show()

#### Ε ΕΡΏΤΗΜΑ ####
N_rand_bits_2 = 10**6
rand_bits_2 = np.random.randint(2, size=(N_rand_bits_2)) #random bits generated
rand_bits_2_mod = rand_bits_2*2*A_bit-A_bit #rand bits modulated (1->A_bit, 0->-A_bit)
t_BER = np.arange(0, 16) #linear space

#EXPERIMENTAL
No_exp, awgn_exp, BER_exp = [], [], []
for i in t_BER:
    No_exp.append(E_b / SNR_dB_lin(i))
    awgn_exp.append(np.random.normal(0, np.sqrt(No_exp[i]), 2*N_rand_bits_2).view(np.complex128))
    output_sign = rand_bits_2_mod + awgn_exp[i].real
    receiv_sign = (output_sign >= 0).astype(int)
    BER_exp.append(np.sum(receiv_sign != rand_bits_2) / N_rand_bits_2)

#THEORETICAL
# BER_theor = scipy.special.erfc(np.sqrt(SNR_dB_lin(t_BER)))
def q_bpam(a):
    return (1.0/math.sqrt(2*math.pi))*scipy.integrate.quad(lambda x: math.exp(-(x**2)/2), a, pow(10,2))[0]
BER_theor = []
for i in t_BER:
    BER_theor.append(q_bpam(np.sqrt(2*SNR_dB_lin(i))))

plt.semilogy(t_BER, BER_exp, color='r', marker='o', markersize=2, linestyle='')
plt.semilogy(t_BER, BER_theor, marker='', linestyle='-', linewidth=1 )
plt.title('Experimental & theoretical BER curve')
plt.xlabel('$E_b/N_0(dB)$');plt.ylabel('BER ($P_e$)')
plt.grid(True)
plt.show()


# t_ber_theor = np.linspace(0, 15, 150, endpoint=False)
# y_ber_theor = []
# y_ber_theor = q_bpam(np.sqrt(2*pow(10, t_ber_theor/10)))
# plt.semilogy(t_ber_theor, y_ber_theor)
# plt.show()