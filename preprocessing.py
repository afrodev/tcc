import pandas as pd

def preprocessing(daysAhead, dataset):
	if daysAhead == 1:
		return prepareDatasetPredict1DayAhead(dataset)
	elif daysAhead == 5:
		return prepareDatasetPredict5DaysAhead(dataset)
	elif daysAhead == 22:
		return prepareDatasetPredict22DaysAhead(dataset)
	else:
		print("invalid days ahead")


# Prepare dataset to predict one day ahead
def prepareDatasetPredict1DayAhead(dataset):
    values = pd.DataFrame(dataset.values)
    dateColumn = values[0] # Date column

    # Open Columns
    openColumn = values[1] # Column Open
    openFirstOffsetColumn = values[1].shift(2) 

    # High Columns
    highColumn = values[2] 
    highFirstOffsetColumn = values[2].shift(2) 

    # Low Columns
    lowColumn = values[3] # Column Open
    lowFirstOffsetColumn = values[3].shift(2)  

    # Close Columns
    closeColumn = values[4]
    closeFirstOffsetColumn = values[4].shift(2) 


    dataframe = pd.concat([dateColumn, 
                           openColumn, openFirstOffsetColumn,
                           highColumn, highFirstOffsetColumn,
                           lowColumn, lowFirstOffsetColumn, 
                           closeColumn, closeFirstOffsetColumn,
                          ], axis=1)

    dataframe.columns = ['date',
                         'open-0', 'open-2',
                         'high-0', 'high-2',
                         'low-0', 'low-2', 
                         'close-0', 'close-2']
    dataframe = dataframe.dropna()
    return dataframe

# Prepare dataset to predict 5 days ahead
def prepareDatasetPredict5DaysAhead(dataset):
    # create a dataset with date, open, open-7 open-6
    values = pd.DataFrame(dataset.values)

    dateColumn = values[0] # Date column

    # Open Columns
    openColumn = values[1] # Column Open
    openOffset6Column = values[1].shift(6) # Open offset 6 column
    openOffset7Column = values[1].shift(7) # Open offset 7 column

    # High Columns
    highColumn = values[2] 
    highOffset6Column = values[2].shift(6) # High offset 6 column
    highOffset7Column = values[2].shift(7) # High offset 7 column

    # Low Columns
    lowColumn = values[3] # Column Open
    lowOffset6Column = values[3].shift(6)  # Low offset 6 column
    lowOffset7Column = values[3].shift(7) # Low offset 7 column

    # Close Columns
    closeColumn = values[4]
    closeOffset6Column = values[4].shift(6) # Close offset 6 column
    closeOffset7Column = values[4].shift(7) # Close offset 7 column


    dataframe = pd.concat([dateColumn, 
                           openColumn, openOffset6Column, openOffset7Column,
                           highColumn, highOffset6Column, highOffset7Column,
                           lowColumn, lowOffset6Column, lowOffset7Column,
                           closeColumn, closeOffset6Column, closeOffset7Column,
                          ], axis=1)

    dataframe.columns = ['date',
                         'open-0', 'open-6', 'open-7', 
                         'high-0', 'high-6', 'high-7', 
                         'low-0', 'low-6', 'low-7',
                         'close-0', 'close-6', 'close-7']
    dataframe = dataframe.dropna()
    return dataframe


# Prepare dataset to predict 22 days ahead
def prepareDatasetPredict22DaysAhead(dataset):
    values = pd.DataFrame(dataset.values)

    dateColumn = values[0] # Date column

    # Open Columns
    openColumn = values[1] # Column Open
    openFirstOffsetColumn = values[1].shift(23) 
    openSecondOffsetColumn = values[1].shift(24) 

    # High Columns
    highColumn = values[2] 
    highFirstOffsetColumn = values[2].shift(23) 
    highSecondOffsetColumn = values[2].shift(24) 

    # Low Columns
    lowColumn = values[3] # Column Open
    lowFirstOffsetColumn = values[3].shift(23)  
    lowSecondOffsetColumn = values[3].shift(24) 

    # Close Columns
    closeColumn = values[4]
    closeFirstOffsetColumn = values[4].shift(23) 
    closeSecondOffsetColumn = values[4].shift(24) 


    dataframe = pd.concat([dateColumn, 
                           openColumn, openFirstOffsetColumn, openSecondOffsetColumn,
                           highColumn, highFirstOffsetColumn, highSecondOffsetColumn,
                           lowColumn, lowFirstOffsetColumn, lowSecondOffsetColumn,
                           closeColumn, closeFirstOffsetColumn, closeSecondOffsetColumn,
                          ], axis=1)

    dataframe.columns = ['date',
                         'open-0', 'open-23', 'open-24', 
                         'high-0', 'high-23', 'high-24', 
                         'low-0', 'low-23', 'low-24',
                         'close-0', 'close-23', 'close-24']
    dataframe = dataframe.dropna()
    return dataframe

