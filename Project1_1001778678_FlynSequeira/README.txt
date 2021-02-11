TO UPLOAD THE RIGHT DATA THE DATA ADDRESS CAN BE CHANGED AT

1. Add "Training" and "Testing" folder to the code.ipynb directory

2. And change the following line of code. 
	Section: PREPROCESSING
	Function: importData(data_category...)


	data = nib.load('/content/drive/My Drive/brain_stuff/' + data_category + '/' + label +
                                                 '/sub' + str(sub) + '/T1_bet_2_0413.nii.gz').get_data()[10:130, 3:168, 41:186]

	TO

	data = nib.load('./' + data_category + '/' + label +
                                                 '/sub' + str(sub) + '/T1_bet_2_0413.nii.gz').get_data()[10:130, 3:168, 41:186]
