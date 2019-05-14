# load and clean-up data
from numpy import nan
from numpy import isnan
from pandas import read_csv
from pandas import to_numeric

# fill missing values with a value at the same time one day ago
def fill_missing(values):
	one_day = 60 * 24
	for row in range(values.shape[0]):
		for col in range(values.shape[1]):
			if isnan(values[row, col]):
				values[row, col] = values[row - one_day, col]

def gen_data():
	# load all data
	dataset = read_csv(r'D:\Project\machine_learning_data\household_power_consumption\household_power_consumption.txt', sep=';', header=0, low_memory=False, infer_datetime_format=True, parse_dates={'datetime':[0,1]}, index_col=['datetime'])
	# mark all missing values
	dataset.replace('?', nan, inplace=True)
	# make dataset numeric
	dataset = dataset.astype('float32')
	# fill missing
	fill_missing(dataset.values)
	# add a column for for the remainder of sub metering
	values = dataset.values
	dataset['sub_metering_4'] = (values[:,0] * 1000 / 60) - (values[:,4] + values[:,5] + values[:,6])
	# save updated dataset
	# dataset.to_csv(r'D:\Project\machine_learning_data\household_power_consumption\household_power_consumption.csv')
	
	# load the new file
	# dataset = read_csv(r'D:\Project\machine_learning_data\household_power_consumption\household_power_consumption.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
	# resample data to daily
	daily_groups = dataset.resample('D')
	daily_data = daily_groups.sum()
	# summarize
	print(daily_data.shape)
	print(daily_data.head())
	# save
	daily_data.to_csv(r'D:\Project\machine_learning_data\household_power_consumption\household_power_consumption_days.csv')
	
def get_data():
	# split a univariate dataset into train/test sets
	def split_dataset(data):
		# split into standard weeks
		train, test = data[1:-328], data[-328:-6]
		# restructure into windows of weekly data
		train = array(split(train, len(train)/7))
		test = array(split(test, len(test)/7))
		return train, test

	# load the new file
	dataset = read_csv('household_power_consumption_days.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
	train, test = split_dataset(dataset.values)

	# # validate train data
	# print(train.shape)
	# print(train[0, 0, 0], train[-1, -1, 0])
	# # validate test
	# print(test.shape)
	# print(test[0, 0, 0], test[-1, -1, 0])

	return train, test
	