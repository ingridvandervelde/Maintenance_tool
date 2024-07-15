import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import warnings
np.random.seed(0)

warnings.simplefilter("ignore")

student_nr = "s4465385"
data_path = "C:/Users/vande/Documents/AssetManagement/s4465385/"


def data_preparation(machine_data, machine_name):
    durations = []
    censored = []
    for index, row in machine_data.iterrows():
        #the first duration is equal to the time
        if index == 0:
            durations.append(row["Time"])
        #the other durations are equal to the time minus the time at the index-1
        else:
            duration = row["Time"] - machine_data.iloc[index-1]["Time"]
            durations.append(duration)
    machine_data["Duration"] = durations

    #censored is Yes for PM and No for failure
    censored = []
    for index, row in machine_data.iterrows():
        if row["Event"] == "PM":
            censored.append("Yes")
        else:
            censored.append("No")
    machine_data["Censored"] = censored
    machine_data = machine_data.sort_values("Duration")

    #when the durations are equal, the values need to be custom sorted
    unique_durations = machine_data['Duration'].unique()
    for duration in unique_durations:
        duration_data = machine_data[machine_data['Duration'] == duration]
        if len(duration_data) > 1:
            #getting the indeces, in order to place them back in duration_data
            pm_indices = duration_data[duration_data['Event'] == 'PM'].index
            failure_indices = duration_data[duration_data['Event'] == 'failure'].index
            sorted_indices = list(pm_indices) + list(failure_indices)
            duration_data = duration_data.loc[sorted_indices]

            machine_data.loc[duration_data.index] = duration_data

    return machine_data

# https://www.prepbytes.com/blog/python/iloc-function-in-python/
# https://www.w3schools.com/python/pandas/ref_df_iterrows.asp
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_values.html
# https://stackoverflow.com/questions/13838405/custom-sorting-in-pandas-dataframe

def create_kaplanmeier_data(prepared_data):
    #first probability
    prepared_data["Probability"] = 1 / len(prepared_data)
    #reset the indeces
    prepared_data.reset_index(drop=True, inplace=True)

    #recalculating the probability when censored is equal to yes for the remaining rows
    for index, row in prepared_data.iterrows():
        if row["Censored"] == "Yes":
            remaining_rows = prepared_data.iloc[index + 1:]
            num_remaining_rows = len(remaining_rows)
            prob_value = row['Probability']
            increment= prob_value / num_remaining_rows
            remaining_rows['Probability'] += increment

    #probability is 0 when censored is  Yes
    for index, row in prepared_data.iterrows():
        if row["Censored"] == "Yes":
            prepared_data.loc[index, 'Probability'] = 0

    #if there are groups of equal durations, then the values should be combined.
    for duration, group in prepared_data.groupby(["Duration"]):
        if len(group) > 1:
            #only No as Yes is 0
            avg_time = group.loc[group["Censored"] == "No", "Time"].mean()
            #taking the first event
            event = group["Event"].iloc[0]
            #value from the iteration
            duration_value = duration
            #is No
            censored = group["Censored"].iloc[0]
            #summing probability
            probability_sum = group.loc[group["Censored"] == "No", "Probability"].sum()
            #putting the new values in the df
            prepared_data.loc[group.index[0]] = [avg_time, event, duration_value, censored, probability_sum]
    #deleting duplicate values, as there should be one combined row
    prepared_data.drop_duplicates(subset="Duration", keep="first", inplace=True)
    prepared_data.reset_index(drop=True, inplace=True)
    #making the new column reliability
    prepared_data["Reliability"] = 1 - prepared_data["Probability"].cumsum()


    return prepared_data
#https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.cumsum.html
#https://www.w3schools.com/python/pandas/ref_df_reset_index.asp
#https://www.datacamp.com/tutorial/pandas-drop-duplicates?utm_source=google&utm_medium=paid_search&utm_campaignid=19589720818&utm_adgroupid=157156373751&utm_device=c&utm_keyword=&utm_matchtype=&utm_network=g&utm_adpostion=&utm_creative=684592138751&utm_targetid=dsa-2218886984100&utm_loc_interest_ms=&utm_loc_physical_ms=1010427&utm_content=&utm_campaign=230119_1-sea~dsa~tofu_2-b2c_3-eu_4-prc_5-na_6-na_7-le_8-pdsh-go_9-na_10-na_11-na&gad_source=1&gclid=EAIaIQobChMIjcCPxsighQMVtKloCR09oQb_EAAYASAAEgKRbvD_BwE
def visualization(KM_data, weibull_data, machine_name):
    plt.plot(KM_data["Duration"], KM_data["Reliability"], label = "Kaplan Meier")
    plt.plot(weibull_data["t"], weibull_data["R_t"], label = "Weibull")
    plt.xlabel("Duration")
    plt.ylabel("Reliability")
    plt.legend()
    plt.title(f'Reliability against Duration for Machine-{machine_name}')
    plt.show()

def meantimebetweenfailure_KM(KM_data):
    #formula of MTBF
    KM_MTBF = (KM_data["Duration"] * KM_data["Probability"]).sum()

    return KM_MTBF

def fit_weibull_distribution(prepared_data):
    l_range = np.linspace(start = 1, stop = 35, num = 35)
    k_range = np.linspace(start = 0.1, stop = 3.5, num = 35)

    data_list = []
    #putting all 1225 combinations in a DF
    for l_value in l_range:
        for k_value in k_range:
            data_list.append((l_value, k_value))

    weib_data = pd.DataFrame(data_list, columns=["Lambda", "Kappa"])

    #filling in each value of the new columns, with a tuple containing censored and duration
    for index, row in prepared_data.iterrows():
        value = (row["Censored"], row["Duration"])
        #making sure the values are in all rows
        weib_data[f'Observation {index}'] = [value] * len(weib_data)

    #looping over the dataset, and assigning functions based on the values in the tuples
    for index, row in weib_data.iterrows():
        for i in range(0, len(prepared_data)):
            if row[f'Observation {i}'][0] == "No":
                kappa = row["Kappa"]
                lamda = row["Lambda"]
                weib_data.loc[index, f'Observation {i}'] = np.log((kappa / lamda) *
                                                              ((row[f'Observation {i}'][1] / lamda) ** (kappa - 1)) *
                                                              math.exp(-(row[f'Observation {i}'][1] / lamda) ** kappa))
            else:
                kappa = row["Kappa"]
                lamda = row["Lambda"]
                weib_data.loc[index, f'Observation {i}'] = np.log(math.exp(-(row[f'Observation {i}'][1] / lamda) ** kappa))
    #summing all observation columns for the log likelihood
    weib_data["Log Likelihood Sum"] = weib_data.iloc[:, 2:].sum(axis =1)

    #max log likelihood, and corresponding k and l
    max_index = weib_data["Log Likelihood Sum"].idxmax()
    k = weib_data.loc[max_index, "Kappa"]
    l = weib_data.loc[max_index, "Lambda"]

    return l, k, weib_data

def meantimebetweenfailure_weibull(l, k):
    Weibull_MTBF = l * math.gamma(1 + (1 / k))

    return Weibull_MTBF

def create_weibull_curve_data(prepared_data, l, k):
    weibull_data = pd.DataFrame()
    #0 - the highest value of duration is added to the column "t"
    weibull_data["t"] = np.arange(0, prepared_data["Duration"].max(), 0.01) #range function does not work for 0.01 steps
    weibull_data["R_t"] = np.exp(-(weibull_data / l) ** k) #math.exp does not work
    return weibull_data

#https://numpy.org/doc/stable/reference/generated/numpy.arange.html
#https://numpy.org/doc/stable/reference/generated/numpy.exp.html
def create_cost_data(weibull_data, l, k, PM_cost, CM_cost, machine_name):
    maintenance_cost = pd.DataFrame()
    maintenance_cost["t"] = weibull_data["t"]
    maintenance_cost["R(t)"] = weibull_data["R_t"]
    maintenance_cost["F(t)"] = 1 - maintenance_cost["R(t)"]
    maintenance_cost["Cost per Cycle"] = (CM_cost * maintenance_cost["F(t)"]) + (PM_cost * maintenance_cost["R(t)"])
    maintenance_cost["Mean Cycle Length"] = (maintenance_cost["R(t)"] * 0.01).cumsum()
    maintenance_cost["Cost rate (t)"] = maintenance_cost["Cost per Cycle"] / maintenance_cost["Mean Cycle Length"]


    plt.plot(maintenance_cost["t"], maintenance_cost["Cost rate (t)"])
    plt.xlabel("Duration")
    plt.ylabel("Cost rate")
    plt.ylim(0,1000)
    plt.title(f"Optimal Maintenance Age of Machine-{machine_name}")
    plt.show()

    #getting the lowest maintenance cost and maintenance age
    best_cost_rate = maintenance_cost["Cost rate (t)"].min()
    index_min_cost = maintenance_cost["Cost rate (t)"].idxmin()
    best_age = maintenance_cost.iloc[index_min_cost]["t"]

    return best_cost_rate, best_age

def CBM_data_preparation(condition_data):
    for i in range(1, len(condition_data)):
        #the increment is the value of the condition minus the condition at one index back
        increment = condition_data.iloc[i]["Condition"] - condition_data.iloc[i-1]["Condition"]
        condition_data.loc[i, "Increments"] = increment
        #keeping only the values greater than 0
    condition_data = condition_data[condition_data["Increments"] > 0]

    return condition_data

def CBM_create_simulations(prepared_condition_data, failure_level, threshold):
    Number_of_simulations = 1000
    #all simulations are added to the list
    simulation_list = []
    simulation_data = pd.DataFrame(columns = ["Duration", "Event"])

    i = 0
    while i < Number_of_simulations: #iterating untill the simulation ends
        condition = 0 #start
        time = 0
        while True:
            time += 1 #increments untill this loops breaks: either at failure or PM
            increment = prepared_condition_data["Increments"].sample().iloc[0] #taking a sample
            condition += increment
            if condition >= failure_level:
                simulation_list.append([time, "failure"])
                break
            elif condition >= threshold:
                simulation_list.append([time, "PM"])
                condition = 0 #condition is set again at 0 when preventive maintenance is carried out
                break
        i += 1
    simulation_data = pd.DataFrame(simulation_list, columns = ["Duration", "Event"])
    return simulation_data
#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sample.html

def CBM_analyze_costs(simulation_data, PM_cost, CM_cost):
    PM_ratio = len(simulation_data[simulation_data["Event"] == "PM"]) / len(simulation_data)
    failure_ratio = len(simulation_data[simulation_data["Event"] == "failure"]) / len(simulation_data)
    PM_cost_CBM = PM_cost * PM_ratio
    CM_cost_CBM = CM_cost * failure_ratio
    Cost_per_cycle = PM_cost_CBM + CM_cost_CBM
    Cycle_length = simulation_data["Duration"].mean()
    Cost_rate = Cost_per_cycle / Cycle_length

    return Cost_rate

def CBM_create_cost_data(prepared_condition_data, PM_cost, CM_cost, failure_level, machine_name):

    #making a list with different thresholds
    threshold_level = np.arange(0.6, 0.98, 0.01)
    #list comprehension
    threshold_list = [t * failure_level for t in threshold_level]

    CBM_cost_data_list = []
    #applying the functions for the differen thresholds and adding it to a DF
    for threshold in threshold_list:
        simulation_data = CBM_create_simulations(prepared_condition_data, failure_level, threshold)
        cost_rate = CBM_analyze_costs(simulation_data, PM_cost, CM_cost)
        CBM_cost_data_list.append([threshold, cost_rate])

    CBM_cost_data = pd.DataFrame(CBM_cost_data_list, columns = ["Threshold", "Cost Rate CBM"])

    plt.plot(CBM_cost_data["Threshold"], CBM_cost_data["Cost Rate CBM"])
    plt.xlabel("Threshold")
    plt.ylabel("Cost Rate for CBM")
    plt.title(f'CBM Cost rate for Different Thresholds for Machine-{machine_name}')
    plt.show()

    CBM_cost_rate = CBM_cost_data["Cost Rate CBM"].min()
    corresponding_index = CBM_cost_data["Cost Rate CBM"].idxmin()
    CBM_threshold = CBM_cost_data.iloc[corresponding_index]["Threshold"]

    return CBM_cost_rate, CBM_threshold
#https://www.datacamp.com/tutorial/python-list-comprehension?utm_source=google&utm_medium=paid_search&utm_campaignid=19589720818&utm_adgroupid=157156373751&utm_device=c&utm_keyword=&utm_matchtype=&utm_network=g&utm_adpostion=&utm_creative=684592138751&utm_targetid=dsa-2218886984100&utm_loc_interest_ms=&utm_loc_physical_ms=1010427&utm_content=&utm_campaign=230119_1-sea~dsa~tofu_2-b2c_3-eu_4-prc_5-na_6-na_7-le_8-pdsh-go_9-na_10-na_11-na&gad_source=1&gclid=EAIaIQobChMIsJyG18yghQMVY2BBAh0VIg4fEAAYASAAEgIzTvD_BwE

def run_analysis():
    #using a for loop to apply all functions to all machines
    machines = range(1,4)
    for machine_name in machines:
        # Data preperation
        machine_data = pd.read_csv(f'{data_path}{student_nr}-Machine-{machine_name}.csv')
        cost_data = pd.read_csv(f'{data_path}{student_nr}-Costs.csv').loc[machine_name - 1]
        PM_cost, CM_cost = cost_data[1], cost_data[2]
        prepared_data = data_preparation(machine_data, machine_name)

#     # Kaplan-Meier estimation
        KM_data = create_kaplanmeier_data(prepared_data)
        MTBF_KM = meantimebetweenfailure_KM(KM_data)
        print(f'The MTBF-KaplanMeier for machine-{machine_name} is: ', "%.2f"%MTBF_KM)

#     # Weibull fitting
        l, k, weib_data = fit_weibull_distribution(prepared_data)
        MTBF_weibull = meantimebetweenfailure_weibull(l, k)
        weibull_data = create_weibull_curve_data(prepared_data, l, k)
        print(f'The MTBF-Weibull for machine-{machine_name} is: ', "%.2f"%MTBF_weibull)

#     #Visualization
        visualization(KM_data, weibull_data, machine_name)

#     #Policy evaluation
        best_age, best_cost_rate = create_cost_data(weibull_data, l, k, PM_cost, CM_cost, machine_name)
        # print('The optimal maintenance age is: ', best_age)
        # print('The best cost rate is: ', best_cost_rate)

        if machine_name == 3: #CBM is applied only for machine 3
            condition_data = pd.read_csv(f'{data_path}{student_nr}-Machine-{machine_name}-condition-data.csv')
            prepared_condition_data = CBM_data_preparation(condition_data)
            failure_level = prepared_condition_data["Condition"].max()
            CBM_threshold, CBM_cost_rate = CBM_create_cost_data(prepared_condition_data, PM_cost, CM_cost, failure_level, machine_name)
            # print('The optimal cost rate under CBM is: ', CBM_cost_rate)
            # print('The optimal CBM threshold is: ', CBM_threshold)

        CM_cost_ = CM_cost / MTBF_weibull #cost for CM formula is from the slides
        if machine_name == 1 or machine_name == 2:
            if k < 1: #CM when k is lower than 1
                print(f"The optimal maintenance policy for machine {machine_name} is: CM")
                print(f"The optimal cost rate for machine {machine_name} is: {'%.2f'%CM_cost_}")
            else: #else preventive maintenance, as machine 1 and 2 don't have CBM data
                print(f"The optimal maintenance policy for machine {machine_name} is: TBM")
                print(f"The optimal cost rate for machine {machine_name} is: {'%.2f'%best_cost_rate}")
                print(f"The savings compared to a pure corrective maintenance policy for {machine_name} are: {'%.2f'%(CM_cost_ - best_cost_rate)}")
        elif machine_name == 3: #comparing all three types of maintenance for machine 3
            if k < 1:
                print(f"The optimal maintenance policy for machine {machine_name} is: CM")
                print(f"The optimal cost rate for machine {machine_name} is: {'%.2f'%CM_cost_}")
            elif best_cost_rate < CBM_cost_rate: #based on the lowest cost
                print(f"The optimal maintenance policy for machine {machine_name} is: TBM")
                print(f"The optimal cost rate for machine {machine_name} is: {'%.2f'%best_cost_rate}")
                print(f"The savings compared to a pure corrective maintenance policy for {machine_name} are: {'%.2f'%(CM_cost_ - best_cost_rate)}")
            else:
                print(f"The optimal maintenance policy for machine {machine_name} is: CBM")
                print(f"The optimal cost rate for machine {machine_name} is: {'%.2f'%CBM_cost_rate}")
                print(f"The savings compared to a pure corrective maintenance policy for {machine_name} are: {'%.2f'%(CM_cost_ - CBM_cost_rate)}")
                print(f"The savings compared to a time-based maintenance policy for {machine_name} are: {'%.2f'%(best_cost_rate - CBM_cost_rate)}")
    return

#https://python.shiksha/tips/limiting-float-upto-2-places/


run_analysis()


