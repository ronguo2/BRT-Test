'''
BRT运行优化

车头时距

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import funs
import para    # input parameter
from geneticalgorithm import geneticalgorithm as ga




def f(x,Veh_w1=0.64,Veh_w2=0.36,isWithinGa=True):
    """
    Model function
    """
    VehDist_L = 2*para.LineLen_D/x[0]       # operation distance

    # fleet size
    WaitTimeVeh_tb = para.Signal_C * (1-para.VehGreen_alpha)**2/(2*(1-para.ArrivalRate_Q/para.LeaveRate_S)) \
                       + (para.ArrivalRate_Q**2/(para.VehGreen_alpha*para.LeaveRate_S)**2)/(2*para.ArrivalRate_Q*(1-para.ArrivalRate_Q/(para.VehGreen_alpha*para.LeaveRate_S)))\
                       -0.65*((para.Signal_C/para.ArrivalRate_Q**2)**(1/3))*((para.ArrivalRate_Q/(para.VehGreen_alpha*para.LeaveRate_S))**(2+5*para.VehGreen_alpha))
    FleetSize_M = VehDist_L / para.VehSpeed_vb + 2 * (para.InterNum_n - 1) * WaitTimeVeh_tb / x[0] + \
                  2 * x[1] * para.Time_stop / x[0] + 2 * (para.InterNum_n + x[1]) * para.Time_lost / x[0]

    # walking time
    Buffer_lb =  0.5*np.abs(para.LineLen_D / (para.InterNum_n -1)-para.LineLen_D / (x[1] -1)) + 0.5 * para.LineLen_D / (x[1] - 1)
    if Buffer_lb > para.lb:
        Buffer_lb = Buffer_lb
    else:
        Buffer_lb = para.lb
    StopPlatLen_time = para.StopPlatLen_so/para.WalkSpeed_vp
    BufferPassen_time = Buffer_lb/para.WalkSpeed_vp
    Crosswalk_time = para.CrosswalkLen_d/para.WalkSpeed_vp
    WaitTimeVeh_tp = para.Signal_C*(1-para.PassenGreen_beta)**2/2
    WalkTime_A = BufferPassen_time + StopPlatLen_time+ Crosswalk_time + WaitTimeVeh_tp

    # waiting time
    WaitTime_W = x[0]/2

    # in-vehicle travel time
    n = int(x[1])-1
    d1=0
    d2 =0
    for i in range(n):
        d1 += 2*i*(n-i)

    for j in range(n):
        d2 += 2*j

    InVehTraTime_T = (1/n)*d1/d2*para.LineLen_D/para.VehSpeed_vb  # in-vehicle travel time

    Utility = (para.wa*WalkTime_A +para.ww*WaitTime_W+para.wt*InVehTraTime_T)*para.value_time + para.wf*para.fare

    # elastic demand
    ElasticPassen = para.Passenger - para.psi*(-1/para.thet)*(np.log(np.exp(-para.thet*Utility)))

    TotalPassen = ElasticPassen*para.LineLen_D

    # equivalent to travel time per passenger
    ChangeLen_Tl = para.value_dist/(TotalPassen*para.value_time)
    ChangeLen_Tm = para.value_hour/(TotalPassen*para.value_time)

    # generalized time costs
    minZ =  (Veh_w1*(ChangeLen_Tl*VehDist_L + ChangeLen_Tm * FleetSize_M) + Veh_w2 *(WalkTime_A + WaitTime_W + InVehTraTime_T ))

    if isWithinGa:
        return minZ
    else:
        return [para.Passenger,ElasticPassen, Veh_w1, Veh_w2, x[0],x[1],
                minZ, VehDist_L,  FleetSize_M,
                WalkTime_A,WaitTime_W, InVehTraTime_T]


def solve_one_ga_para_seting():
    """
    Solving models using elite genetic algorithms
    """
    varbound = np.array([[0, 1], [5, 50]])    # Variable range
    vartype = np.array([['real'], ['int']])   # Variable type

    # GA Parameters
    algorithm_param = {'max_num_iteration': para.max_ga_iter,
                       'population_size': para.ga_pop_size,
                       'mutation_probability': 0.1,
                       'elit_ratio': 0.01,
                       'crossover_probability': 0.5,
                       'parents_portion': 0.5,
                       'crossover_type': 'uniform',
                       'max_iteration_without_improv': None}
    model = ga(function=f, dimension=2,
               variable_type_mixed=vartype,
               variable_boundaries=varbound,
               algorithm_parameters=algorithm_param,
               convergence_curve=True)

    # Methods and Outputs
    model.run()
    ans = f(model.best_variable, isWithinGa=False)

    return ans

def Test_lamda(_Test_Save_Folder_Name: str):
    """
    Optimal results for different demand densities and saved in excel
    """
    df = pd.DataFrame(columns=['PotentialPassen','ElasticPassen', 'w1', 'w2', 'opt_headway','opt_StopNumber','GeneralizedCosts', 'VehDist_L', 'FleetSize_M','WalkTime_A', 'WaitTime_W', 'InVehTraTime_T'])

    # actual test
    # for i in range(20, 201, 20):
    #     para.Passenger = i
    #     ans2 = f(0.167, 16, 0.64, 0.36)
    #     df2 = pd.DataFrame(np.array(ans2).reshape(1, -1), columns=df.columns)
    #     df = pd.concat([df, df2])

    # optimal test
    for i in range(20, 201, 20):
        para.Passenger = i
        ans = solve_one_ga_para_seting()
        df1 = pd.DataFrame(np.array(ans).reshape(1, -1), columns=df.columns)
        df = pd.concat([df, df1])
#
    df.to_excel('result_optimal_test.xlsx', encoding="utf-8", index=None)
    plt.rcParams['font.sans-serif'] = ['Times New Roman']

if __name__ == "__main__":
    Test_lamda("result_actual")

print('finish')