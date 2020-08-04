import argparse
import struct
import traceback
import numpy as np
from datetime import datetime

from prettytable import PrettyTable
from prettytable import from_csv
from prettytable import from_html

import matplotlib.pyplot as plt
import matplotlib.colors as colors

######################
# plt.ion()
# plt.show()
# fig=plt.figure()
# frq=np.arange(0,100,1)
# t=np.arange(100) 
## Color Mesh a reproduzir 1's e 0's da matriz resultante depois de aplicado o threshold 
# plt.pcolormesh(frq,t,S,norm=colors.SymLogNorm(linthresh=0.5, linscale=0.03,vmin=0, vmax=1))
# plt.draw()
# fig.canvas.flush_events()
# ##plt.clf()
# waitforEnter()
########################

# Cria a tabela
table = PrettyTable(["Métrica", "Minimo", "Máximo", "Média", "Mediana", "Percentil 75%","Percentil 90%","Percentil 95%","Percentil 99%"])

# Alinha as colunas
table.align["Métrica"] = "l"
table.align["Minimo"] = "l"
table.align["Máximo"] = "r"
table.align["Média"] = "r"
table.align["Mediana"] = "r"
table.align["Percentil 75%"] = "r"
table.align["Percentil 90%"] = "r"
table.align["Percentil 95%"] = "r"
table.align["Percentil 99%"] = "r"

# Deixa um espaço entre a borda das colunas e o conteúdo (default)
table.padding_width = 1

sample_number = 0
fn=None
auxiliar = False
timestampBegin = ''
timestampEnd = ''
timeSize = 0
beginings = []
tmp = []

def window_analysis(observation_window):

    global sample_number
    global fn
    global beginings
    global timestampBegin
    global timestampEnd
    global tmp

    x = [a[0] for a in observation_window]
    y = [a[1] for a in observation_window]
    z = [a[2] for a in observation_window]
    w = [a[4] for a in observation_window]
    with_data = False

    print("X:",x)

    ob = []
    result = []

    if auxiliar:

        begin = datetime.strptime(timestampBegin,'%Y-%m-%d %H:%M:%S.%f')
        end = datetime.strptime(timestampEnd,'%Y-%m-%d %H:%M:%S.%f')
        for data in observation_window:
            print(data)
            sample_number += 1
            time_data = datetime.fromtimestamp(data[3])

            begin = datetime.strptime(timestampBegin,'%Y-%m-%d %H:%M:%S.%f')
            end = datetime.strptime(timestampEnd,'%Y-%m-%d %H:%M:%S.%f')
            
            if time_data >= begin and time_data <= end:
                print("Begin: ",str(begin),"| End: ", str(end), "| Atual: ", time_data ) 
                ob.append(data)
                tmp.append(sample_number)
                with_data = True
            
            elif time_data > end :  
                print("MAIOR | Begin: ",str(begin),"| End: ", str(end), "| Atual: ", time_data )
                timestamp = fn.readline()

                if len(timestamp) >= timeSize:
                
                    timestampBegin = timestamp.split(" ")[0]+" "+timestamp.split(" ")[1]
                    timestampEnd = timestamp.split(" ")[2]+" "+timestamp.split(" ")[3]
                
                print(tmp[0], tmp[-1])
                beginings.append(tmp[0])
                beginings.append(tmp[-1])
                tmp = []

        if with_data:
            x = [a[0] for a in ob]
            print(x)
            y = [a[1] for a in ob]
            z = [a[2] for a in ob]
            w = [a[4] for a in ob]

        result.append( [ '{:.2f}'.format(np.min(x)),'{:.2f}'.format(np.max(x)),'{:.2f}'.format(np.mean(x)),'{:.2f}'.format(np.median(x)), '{:.2f}'.format(np.percentile(x,75)), '{:.2f}'.format(np.percentile(x,90)), '{:.2f}'.format(np.percentile(x,95)), '{:.2f}'.format(np.percentile(x,99)) ] )
        result.append( [ '{:.2f}'.format(np.min(y)),'{:.2f}'.format(np.max(y)),'{:.2f}'.format(np.mean(y)),'{:.2f}'.format(np.median(y)), '{:.2f}'.format(np.percentile(y,75)), '{:.2f}'.format(np.percentile(y,90)), '{:.2f}'.format(np.percentile(y,95)), '{:.2f}'.format(np.percentile(y,99)) ] )
        result.append( [ '{:.2f}'.format(np.min(z)),'{:.2f}'.format(np.max(z)),'{:.2f}'.format(np.mean(z)),'{:.2f}'.format(np.median(z)), '{:.2f}'.format(np.percentile(z,75)), '{:.2f}'.format(np.percentile(z,90)), '{:.2f}'.format(np.percentile(z,95)), '{:.2f}'.format(np.percentile(z,99)) ] )
    
    return result
    
def main() :

    # input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-f'  , '--file', help='Filename', required=True)
    parser.add_argument('-a'  , '--auxiliar', help='Auxiliar File')
    parser.add_argument('-ws', '--windowSize', type=int, help='the size of each observation window ( number of samples ).', default=50)
    parser.add_argument('-wo', '--windowOffset', type=int, help='number of shifted samples.', default=50)
    args = parser.parse_args()

    # The output file from the phase 1
    # Set of metrics for each sample
    input_file = args.file
    
    # The exact number of samples for each Observation Window
    window_size = args.windowSize
    
    # The number of offset samples
    window_offset = args.windowOffset

    # initialization of the observation window
    observation_window = []

    # The number of the current Observation Window
    window_number = 1

    # The initialization of the number of samples
    sample = 0

    # open the file
    fb=open(input_file,'rb')

    # open the file to write the data
    out=open("data/"+args.file.split('/')[-1].replace('_phase_1.dat','_phase_2.dat'),'wb')

    data = []

    global auxiliar

    if args.auxiliar:
        auxiliar = True

        global fn
        fn=open(args.auxiliar,'r')

        timestamp = fn.readline()
        lista = timestamp.split(' ')
        
        global timestampBegin
        timestampBegin = lista[0]+" "+lista[1]

        global timestampEnd
        timestampEnd = lista[2]+" "+lista[3].replace('\n','')
        
        global timeSize
        timeSize = len(timestamp)

    try:

        while True:
            
            # read each line of the file
            record = fb.read(40)

            # break when the line was empty
            if(len(record) < 40): break

            # unpack the line
            line = list(struct.unpack('=5d', record))

            # The observation window will have de size of window size
            if sample < window_size:
                # append the sample to the window
                observation_window.append(line)
                
                # increase the number of samples inside the window
                sample += 1
            else:
                # jump to the next position in the file
                fb.seek( 40*window_offset*window_number )
                
                # increase the number of windows
                window_number += 1
                
                # number of samples for each observation window
                sample = 0

                # analyse the window
                # remove the first 3 measurements
                if window_number >= 3:
                    
                    data.append(window_analysis(observation_window))
                    #out.write(struct.pack("4d",*data))
                
                # reset the observation window
                observation_window = []  
    except BaseException as e:
        print(repr(e))
        print(traceback.format_exc())
    finally:        
        fb.close()
        out.close()

    MFIC_minimo = []
    MFIC_maximo = []
    MFIC_media = []
    MFIC_mediana = []
    MFIC_percentil75 = []
    MFIC_percentil90 = []
    MFIC_percentil95 = []
    MFIC_percentil99 = []

    # Diferença entre a posição da última Freq. Ativa e a 1ª
    DFU_minimo = []
    DFU_maximo = []
    DFU_media = []
    DFU_mediana = []
    DFU_percentil75 = []
    DFU_percentil90 = []
    DFU_percentil95 = []
    DFU_percentil99 = []
    
    # Média de Frequências ativas consecutivas
    MFAC_minimo = []
    MFAC_maximo = []
    MFAC_media = []
    MFAC_mediana = []
    MFAC_percentil75 = []
    MFAC_percentil90 = []
    MFAC_percentil95 = []
    MFAC_percentil99 = []

    print(data)
    for d in data:
        print("Observation Window")
        print(*d, sep='\n')
        
        # Média de Frequências inativas consecutivas
        MFIC_minimo.append( float(d[0][0]) )
        MFIC_maximo.append( float(d[0][1]) )
        MFIC_media.append( float(d[0][2]) )
        MFIC_mediana.append( float(d[0][3]) )
        MFIC_percentil75.append( float(d[0][4]) )
        MFIC_percentil90.append( float(d[0][5]) )
        MFIC_percentil95.append( float(d[0][6]) )
        MFIC_percentil99.append( float(d[0][7]) )

        # Diferença entre a posição da última Freq. Ativa e a 1ª
        DFU_minimo.append( float(d[1][0]) )
        DFU_maximo.append( float(d[1][1]) )
        DFU_media.append( float(d[1][2]) )
        DFU_mediana.append( float(d[1][3]) )
        DFU_percentil75.append( float(d[1][4]) )
        DFU_percentil90.append( float(d[1][5]) )
        DFU_percentil95.append( float(d[1][6]) )
        DFU_percentil99.append( float(d[1][7]) )
        
        # Média de Frequências ativas consecutivas
        MFAC_minimo.append( float(d[2][0]) )
        MFAC_maximo.append( float(d[2][1]) )
        MFAC_media.append( float(d[2][2]) )
        MFAC_mediana.append( float(d[2][3] ) )
        MFAC_percentil75.append( float(d[2][4]) )
        MFAC_percentil90.append( float(d[2][5]) )
        MFAC_percentil95.append( float(d[2][6]) )
        MFAC_percentil99.append( float(d[2][7]) )

    fig = plt.figure()
    
    ax = fig.add_subplot(421)  
    ax.yaxis.set_ticks(np.arange( min(MFIC_minimo), max(MFIC_minimo), 1.0))
    ax.title.set_text("Média de Frequências inativas consecutivas | MÍNIMO ")
    ax.plot(MFIC_minimo, color='k')

    bx = fig.add_subplot(422)
    bx.title.set_text("Média de Frequências inativas consecutivas | MÁXIMO ")
    bx.yaxis.set_ticks(np.arange( min(MFIC_maximo), max(MFIC_maximo), 7.5))
    bx.plot(MFIC_maximo, color='k')

    cx = fig.add_subplot(423)
    cx.title.set_text("Média de Frequências inativas consecutivas | MEDIA ")
    cx.yaxis.set_ticks(np.arange( min(MFIC_media), max(MFIC_media), 5.0))
    cx.plot(MFIC_media, color='k')

    dx = fig.add_subplot(424)
    dx.title.set_text("Média de Frequências inativas consecutivas | MEDIANA ")
    dx.yaxis.set_ticks(np.arange( min(MFIC_mediana), max(MFIC_mediana), 2.0))
    dx.plot(MFIC_mediana, color='k')

    ex = fig.add_subplot(425)
    ex.title.set_text("Média de Frequências inativas consecutivas | PERCENTIL 75% ")
    ex.yaxis.set_ticks(np.arange( min(MFIC_percentil75), max(MFIC_percentil75), 5.0))
    ex.plot(MFIC_percentil75, color='k')

    fx = fig.add_subplot(426)
    fx.title.set_text("Média de Frequências inativas consecutivas | PERCENTIL 90% ")
    fx.yaxis.set_ticks(np.arange( min(MFIC_percentil90), max(MFIC_percentil90), 10.0))
    fx.plot(MFIC_percentil90, color='k')

    gx = fig.add_subplot(427)
    gx.title.set_text("Média de Frequências inativas consecutivas | PERCENTIL 95% ")
    gx.yaxis.set_ticks(np.arange( min(MFIC_percentil95), max(MFIC_percentil95), 10.0))
    gx.plot(MFIC_percentil95, color='k')

    hx = fig.add_subplot(428)
    hx.title.set_text("Média de Frequências inativas consecutivas | PERCENTIL 99% ")
    hx.yaxis.set_ticks(np.arange( min(MFIC_percentil99), max(MFIC_percentil99), 10.0))
    hx.plot(MFIC_percentil99, color='k')

    #################################
    fig0 = plt.figure()
    
    ix = fig0.add_subplot(421)
    ix.title.set_text("Diferença entre a posição da última Freq. Ativa e a 1ª | MÍNIMO ")
    ix.yaxis.set_ticks(np.arange( min(DFU_minimo), max(DFU_minimo), 1.0))
    ix.plot(DFU_minimo, color='k')

    jx = fig0.add_subplot(422)
    jx.title.set_text("Diferença entre a posição da última Freq. Ativa e a 1ª | MÁXIMO ")
    jx.yaxis.set_ticks(np.arange( min(DFU_maximo), max(DFU_maximo), 7.5))
    jx.plot(DFU_maximo, color='k')

    kx = fig0.add_subplot(423)
    kx.title.set_text("Diferença entre a posição da última Freq. Ativa e a 1ª | MEDIA ")
    kx.yaxis.set_ticks(np.arange( min(DFU_media), max(DFU_media), 5.0))
    kx.plot(DFU_media, color='k')

    lx = fig0.add_subplot(424)
    lx.title.set_text("Diferença entre a posição da última Freq. Ativa e a 1ª | MEDIANA ")
    lx.yaxis.set_ticks(np.arange( min(DFU_mediana), max(DFU_mediana), 2.0))
    lx.plot(DFU_mediana, color='k')

    mx = fig0.add_subplot(425)
    mx.title.set_text("Diferença entre a posição da última Freq. Ativa e a 1ª | PERCENTIL 75% ")
    mx.yaxis.set_ticks(np.arange( min(DFU_percentil75), max(DFU_percentil75), 5.0))
    mx.plot(DFU_percentil75, color='k')

    nx = fig0.add_subplot(426)
    nx.title.set_text("Diferença entre a posição da última Freq. Ativa e a 1ª | PERCENTIL 90% ")
    nx.yaxis.set_ticks(np.arange( min(DFU_percentil90), max(DFU_percentil90), 10.0))
    nx.plot(DFU_percentil90, color='k')

    ox = fig0.add_subplot(427)
    ox.title.set_text("Diferença entre a posição da última Freq. Ativa e a 1ª | PERCENTIL 95% ")
    ox.yaxis.set_ticks(np.arange( min(DFU_percentil95), max(DFU_percentil95), 10.0))
    ox.plot(DFU_percentil95, color='k')

    px = fig0.add_subplot(428)
    px.title.set_text("Diferença entre a posição da última Freq. Ativa e a 1ª | PERCENTIL 99% ")
    px.yaxis.set_ticks(np.arange( min(DFU_percentil99), max(DFU_percentil99), 10.0))
    px.plot(DFU_percentil99, color='k')

    #################################
    fig1 = plt.figure()

    qx = fig1.add_subplot(421)
    qx.title.set_text("Média de Frequências ativas consecutivas | MÍNIMO ")
    qx.yaxis.set_ticks(np.arange( min(MFAC_minimo), max(MFAC_minimo), 1.0))
    qx.plot(MFAC_minimo, color='k')

    rx = fig1.add_subplot(422)
    rx.title.set_text("Média de Frequências ativas consecutivas | MÁXIMO ")
    rx.yaxis.set_ticks(np.arange( min(MFAC_maximo), max(MFAC_maximo), 7.5))
    rx.plot(MFAC_maximo, color='k')

    sx = fig1.add_subplot(423)
    sx.title.set_text("Média de Frequências ativas consecutivas | MEDIA ")
    sx.yaxis.set_ticks(np.arange( min(MFAC_media), max(MFAC_media), 5.0))
    sx.plot(MFAC_media, color='k')

    tx = fig1.add_subplot(424)
    tx.title.set_text("Média de Frequências ativas consecutivas | MEDIANA ")
    tx.yaxis.set_ticks(np.arange( min(MFAC_mediana), max(MFAC_mediana), 2.0))
    tx.plot(MFAC_mediana, color='k')

    ux = fig1.add_subplot(425)
    ux.title.set_text("Média de Frequências ativas consecutivas | PERCENTIL 75% ")
    ux.yaxis.set_ticks(np.arange( min(DFU_percentil75), max(DFU_percentil75), 5.0))
    ux.plot(MFAC_percentil75, color='k')

    xx = fig1.add_subplot(426)
    xx.title.set_text("Média de Frequências ativas consecutivas | PERCENTIL 90% ")
    xx.yaxis.set_ticks(np.arange( min(MFAC_percentil90), max(MFAC_percentil90), 10.0))
    xx.plot(MFAC_percentil90, color='k')

    yx = fig1.add_subplot(427)
    yx.title.set_text("Média de Frequências ativas consecutivas | PERCENTIL 95% ")
    yx.yaxis.set_ticks(np.arange( min(MFAC_percentil95), max(MFAC_percentil95), 10.0))
    yx.plot(MFAC_percentil95, color='k')

    zx = fig1.add_subplot(428)
    zx.title.set_text("Média de Frequências ativas consecutivas | PERCENTIL 99% ")
    zx.yaxis.set_ticks(np.arange( min(MFAC_percentil99), max(MFAC_percentil99), 10.0))
    zx.plot(MFAC_percentil99, color='k')

    plt.show()
    print("Beginnings: ",beginings)

if __name__ == '__main__':
	main()