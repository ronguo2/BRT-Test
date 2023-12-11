"""
input parameter
"""

root_folder = "D:\****\\"  # file path

max_ga_iter = 1000
ga_pop_size = 50

invalidNum = -9999999
LineLen_D =10
CrosswalkLen_d = 0.02
StopPlatLen_so = 0.025

VehGreen_alpha = 0.45
PassenGreen_beta = 0.31
value_time = 20

Signal_C = 3/60   #3min

ArrivalRate_Q = 0.05*60*60
LeaveRate_S = 0.4*60*60

Time_lost = 2/60/60
Time_stop = 20/60/60

value_dist = 2
value_hour = 40

thet = 0.1
psi = 0.5

res=1
wa = res
ww = res
wt = res
wf = res
ParkingLine_lp = 0.07

acceleration = 0.5*12960   #km/h^2

VehSpeed_vb = 25
WalkSpeed_vp =5

InterNum_n = 16
fare = 2

lb = VehSpeed_vb**2/(2*acceleration)
