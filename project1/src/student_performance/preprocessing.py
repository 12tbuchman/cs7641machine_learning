import csv
import random
from sklearn.preprocessing import OneHotEncoder

TEST_SIZE = 0.3
RANDOM_STATE = 42

path_to_datasets = "../../../datasets/student_performance/"

random.seed(RANDOM_STATE)

global data, num_grades, cat_grades, pf_grades

def transpose(mat):
	mat_t = [[mat[j][i] for j in range(len(mat))] for i in range(len(mat[0]))]
	return mat_t

def convert_char_mat_to_int_mat(char_mat):
	int_mat = []
	for char_list in char_mat:
		int_mat.append([int(c) for c in char_list])
	return int_mat

def convert_num_grades_to_grades(passing):
	global num_grades, cat_grades, pf_grades
	for grade in num_grades:
		if grade < 10:
			cat_grades.append('F')
			pf_grades.append(0)
		elif grade < 12:
			cat_grades.append('D')
			pf_grades.append(0 if ord(passing) < ord('D') else 1)
		elif grade < 14:
			cat_grades.append('C')
			pf_grades.append(0 if ord(passing) < ord('C') else 1)
		elif grade < 16:
			cat_grades.append('B')
			pf_grades.append(0 if ord(passing) < ord('B') else 1)
		else:
			cat_grades.append('A')
			pf_grades.append(1)

def load_dataset(use_math=True, setup='A', passing='F'):
	global data, num_grades, cat_grades, pf_grades
	if use_math:
		filename = "student-mat.csv"
	else:
		filename = "student-por.csv"

	data = []
	num_grades = []
	cat_grades = []
	pf_grades = []
	with open(path_to_datasets + filename) as f:
		reader = csv.reader(f, delimiter=';')
		for line in reader:
			data.append(line)
	data = data[1:]  # remove feature names
	data_t = transpose(data)
	labels = data_t[-1]
	num_grades = convert_char_mat_to_int_mat([labels])[0]
	convert_num_grades_to_grades(passing)
	numeric_features_flag = [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
	if setup == 'A':
		data_t = data_t[:32]
	elif setup == 'B':
		data_t = data_t[:32]
		data_t.pop(30)
		numeric_features_flag = numeric_features_flag[:31]
	elif setup == 'C':
		data_t = data_t[:30]
		numeric_features_flag = numeric_features_flag[:30]
	data_t_numeric = []
	data_t_categorical = []
	for index, flag in enumerate(numeric_features_flag):
		if flag:
			data_t_numeric.append(data_t[index])
		else:
			data_t_categorical.append(data_t[index])
	data_t_numeric = convert_char_mat_to_int_mat(data_t_numeric)
	data_categorical = transpose(data_t_categorical)
	enc = OneHotEncoder(dtype=int, drop='if_binary', sparse=False)
	enc.fit(data_categorical)
	data_categorical_expanded = enc.transform(data_categorical)
	data_t_categorical_expanded = transpose(data_categorical_expanded)
	data_t_expanded = data_t_categorical_expanded + data_t_numeric
	data = transpose(data_t_expanded)
	return data, num_grades, cat_grades, pf_grades

def sample_dataset(grades_type, num_samples=649):
	global data, num_grades, cat_grades, pf_grades
	if not 0 < num_samples <= len(data):
		print("Function sample_balanced_dataset called with invalid outcome_samples parameter.")
		print("samples must be in the range 0 < outcome_samples < len(data)")
		return
	grades_copy = []
	if grades_type == 'n':
		grades_copy = num_grades.copy()
	elif grades_type == 'c':
		grades_copy = cat_grades.copy()
	elif grades_type == 'p':
		grades_copy = pf_grades.copy()
	else:
		grades_copy = pf_grades.copy()
	data_copy = data.copy()
	data_zipped = list(zip(data_copy, grades_copy))
	random.shuffle(data_zipped)
	data_shuffled, grades_shuffled = zip(*data_zipped)
	sample_data = data_shuffled[:num_samples]
	sample_grades = grades_shuffled[:num_samples]
	return sample_data, sample_grades
