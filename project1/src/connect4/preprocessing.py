import csv
import random
from sklearn.preprocessing import OneHotEncoder

TEST_SIZE = 0.4
RANDOM_STATE = 42

path_to_datasets = "../../../datasets/connect4/"

random.seed(RANDOM_STATE)

global data, outcomes, win_data, loss_data, draw_data

def transpose(mat):
	mat_t = [[mat[j][i] for j in range(len(mat))] for i in range(len(mat[0]))]
	return mat_t

def load_dataset():
	filename = "connect-4.data"
	global data
	data = []
	with open(path_to_datasets + filename) as f:
		reader = csv.reader(f, delimiter=',')
		for line in reader:
			data.append(line)
	data_t = transpose(data)
	global outcomes
	outcomes = data_t[-1]
	for i, result in enumerate(outcomes):
		outcomes[i] = 0 if result == "win" else 1 if result == "loss" else 2
	data_t.pop(-1)
	data = transpose(data_t)
	enc = OneHotEncoder(drop='first', dtype=int, sparse=False).fit(data)
	data = enc.transform(data)
	global win_data, loss_data, draw_data
	win_data = []
	loss_data = []
	draw_data = []
	for index, outcome in enumerate(outcomes):
		if outcome == 0:
			win_data.append(data[index])
		elif outcome == 1:
			loss_data.append(data[index])
		else:
			draw_data.append(data[index])

def sample_unbalanced_dataset(samples):
	if not 0 < samples <= len(data):
		print("Function sample_dataset called with invalid total_samples parameter.")
		print("samples must be in the range 0 < total_samples < len(data)")
		return
	sample_data = data
	sample_outcomes = outcomes
	data_outcomes = list(zip(sample_data, sample_outcomes))
	random.shuffle(data_outcomes)
	sample_data, sample_outcomes = zip(*data_outcomes)
	sample_data = sample_data[:samples]
	sample_outcomes = sample_outcomes[:samples]
	return sample_data, sample_outcomes

def sample_balanced_dataset(samples=19347):
	outcome_samples = int(samples / 3)
	if not 0 < outcome_samples <= len(draw_data):
		print("Function sample_balanced_dataset called with invalid outcome_samples parameter.")
		print("samples must be in the range 0 < outcome_samples < len(data)")
		return
	random.shuffle(win_data)
	random.shuffle(loss_data)
	random.shuffle(draw_data)
	sample_data = win_data[:outcome_samples] + loss_data[:outcome_samples] + draw_data[:outcome_samples]
	sample_outcomes = [0] * outcome_samples + [1] * outcome_samples + [2] * outcome_samples
	data_outcomes = list(zip(sample_data, sample_outcomes))
	random.shuffle(data_outcomes)
	sample_data, sample_outcomes = zip(*data_outcomes)
	return sample_data, sample_outcomes
