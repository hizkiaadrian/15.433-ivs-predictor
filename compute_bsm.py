print("This code will rerun the computation of Black-Scholes parameters (IT WILL TAKE A LONG TIME). Are you sure? (Y/N)")
doSave = input()

if doSave == 'Y':
    DATA_FOLDER = 'data'

    ## Get BSM parameters

    import pandas as pd
    from datetime import datetime
    from utils import bsm

    calls = pd.DataFrame(columns = ['date', 'C', 'S', 'K', 'r', 't', 'sigma'])
    puts = pd.DataFrame(columns = ['date', 'P', 'S', 'K', 'r', 't', 'sigma'])

    for year in range(2006,2021):
        temp_c, temp_p = bsm.get_parameters_df(year, data_folder=DATA_FOLDER, verbose=True)
        
        calls = calls.append(temp_c)
        puts = puts.append(temp_p)
        
    calls.reset_index(drop=True, inplace=True)
    puts.reset_index(drop=True, inplace=True)

    try:
        calls['date'] = calls['date'].apply(lambda x: datetime.strftime(datetime.strptime(x, '%d-%m-%y'), '%Y-%m-%d'))
        puts['date'] = puts['date'].apply(lambda x: datetime.strftime(datetime.strptime(x, '%d-%m-%y'), '%Y-%m-%d'))
    except:
        pass

    calls.to_feather('calls_bsm.feather')
    puts.to_feather('puts_bsm.feather')
    
else:
    return