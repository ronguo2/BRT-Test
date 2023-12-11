"""
Drawing, the default parameters of the test as an example
"""
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

import io
from PIL import Image

def Test_lamda(_Test_Save_Folder_Name: str):
    # rc('mathtext', default='regular')
    plt.rcParams['font.sans-serif'] = ['Times New Roman']

    # plt.rc['font.size'] = ['16']
    size =12
    smallsize = 12
    bigsize = 14
    plt.rc('legend',fontsize=size)
    res = 0
    res1 =26
    res2 =13

    # todo:plot variable

    data_default=pd.read_excel('Test Data_default.xlsx')   # file path

    # retrieve value
    Passen=data_default.iloc[2+res:12+res,0]

    headway_opt=data_default.iloc[2+res:12+res,4]
    headway_falt = data_default.iloc[2+res1:12+res1,4]
    headway_peak = data_default.iloc[2+res2:12+res2,4]

    totalcost_opt = data_default.iloc[2 + res:12 + res, 6]
    totalcost_falt = data_default.iloc[2 + res1:12 + res1, 6]
    totalcost_peak = data_default.iloc[2 + res2:12 + res2, 6]

    stopnum_opt = data_default.iloc[2 + res:12 + res, 5]
    stopnum_flatAndpeak = data_default.iloc[2 + res2:12 + res2, 5]


    # drawings
    fig= plt.figure(figsize=(12, 5))
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.4)

    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 4)


    ax1.plot(Passen, totalcost_opt, '-',marker='o', label=r'Optimal')
    ax1.plot(Passen, totalcost_peak, '--', marker='*', label=r'Actual peak')
    ax1.plot(Passen, totalcost_falt, '-.', marker='s', label=r'Actual off-peak')

    ax2.plot(Passen, headway_opt, '-', marker='o', label=r'Optimal')
    ax2.plot(Passen, headway_peak, '--', marker='*', label=r'Actual peak')
    ax2.plot(Passen, headway_falt, '-.', marker='s', label=r'Actual off-peak')

    ax3.plot(Passen, stopnum_opt, '-', marker='o', label=r'Optimal')
    ax3.plot(Passen, stopnum_flatAndpeak, '--',marker='*', label=r'Actual')


    ax1.legend(loc=0)
    ax2.legend(loc=0)
    ax3.legend(loc=0)

    ax2.set_ylabel(r'$\it{H}$  (h)',fontsize = smallsize)

    ax1.set_xlabel(r'$\lambda$$_{0}$  (pax/h/km)', fontsize=smallsize)
    ax1.set_ylabel(r'$\it{Z}$  (h)', fontsize=smallsize)

    ax3.set_xlabel(r'$\lambda$$_{0}$  (pax/h/km)', fontsize=smallsize)
    ax3.set_ylabel(r' $\it{N}$', fontsize=smallsize)


    ax1.set_xticks(np.linspace(20, 200, 10))
    ax2.set_xticks(np.linspace(20, 200, 10))
    ax3.set_xticks(np.linspace(20, 200, 10))
    ax3.set_yticks([16, 21, 25])



    ax1.tick_params(axis='both',  labelsize=size)
    ax2.tick_params(axis='both', labelsize=size)
    ax3.tick_params(axis='both', labelsize=size)


    ax1.set_title('(a) Generalized cost ', fontsize=bigsize)
    ax2.set_title('(b) Headway', fontsize=bigsize)
    ax3.set_title('(c) Number of BRT stops', fontsize=bigsize)

    # output in tiff format
    png1=io.BytesIO()
    fig.savefig(png1, dpi=600, format='png', bbox_inches='tight', pad_inches=0.1)
    png2 = Image.open(png1)
    png2.save("Fig 4.tiff")
    png1.close()


    plt.show()


if __name__ == "__main__":
    Test_lamda("lambda_result")

print('finish')


