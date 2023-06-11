import numpy as np
import csv
import math
import random
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

def rangeOfVoice(voice):
    solution = []
    solution.append(0)
    for pitch in voice:
        if pitch not in solution and pitch != 0:
            solution.append(pitch)

    solution = sorted(solution)
    return solution       

def changeToVector(midi_note, voiceNotes):
    if(midi_note == 0):
        return [0,0,0,0,0]
    
    #48 is MIDI value of middle note
    note = np.mod(midi_note - 48, 12) + 1
    chroma_angle = (chroma[note - 1]) * (360/12)
    c5_angle = (c5[note -1]) * (360/12)
    chroma_x = radius_chroma * math.sin(math.radians(chroma_angle))
    chroma_y = radius_chroma *  math.cos(math.radians(chroma_angle))
    c5_x = radius_c5 *  math.sin(math.radians(c5_angle))
    c5_y = radius_c5 *  math.cos(math.radians(c5_angle))
    n = midi_note - 69
    fx = pow(2,(n/12))*440
    max_note = max(voiceNotes)
    min_note = min(voiceNotes[1:])
    max_p = 2 * math.log2(pow(2,((max_note - 69)/12)) * 440)
    min_p = 2 * math.log2(pow(2,((min_note - 69)/12)) * 440)
    pitch = 2 * math.log2(fx) - max_p + (max_p - min_p)/2
    u = [pitch, chroma_x, chroma_y, c5_x,c5_y]
    return u

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_d(x):
    sigmoid_x = 1 / (1 + np.exp(-x))
    sigmoid_derivative_x = sigmoid_x * (1 - sigmoid_x)
    return sigmoid_derivative_x

def softmax(x):
    e_x = np.exp(x) 
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def forward_propagation(x):
    hidden_layer_output = sigmoid(np.dot(x, weights_hidden) + biases_hidden)
    output_layer_output = np.dot(hidden_layer_output, weights_output) + biases_output
    return softmax(output_layer_output), hidden_layer_output


file = open(f'/Users/amir0/Documents/NN/BACHSMASTERPIECE/F.txt')
csvreader = csv.reader(file, delimiter='\t')
voice1 = np.array([int(row[0]) for row in csvreader])
fugue3 = voice1.copy()
#chroma circle: first note is C
chroma = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
radius_chroma = 1
# c5 
c5 = [1, 8, 3, 10, 5, 12, 7, 2, 9, 4, 11, 6]
radius_c5 = 1
voice1Notes = rangeOfVoice(voice1)
transformed_data = []
for midi_note in fugue3:
    u = changeToVector(midi_note, voice1Notes)
    transformed_data.append(u)

#x_train = transformed_data[:-1].copy()
#window size of 16
x_train = np.zeros((len(voice1) - 24, 24*5))
for i in range(len(voice1) - 24):
    x = transformed_data[i:i+24]
    x_train[i] = [element for sublist in x for element in sublist]
x_train = x_train[:-1]

target = np.zeros((len(voice1) - 24, len(voice1Notes)))
index = 0
for pitch in fugue3[24:]:
    pos = voice1Notes.index(pitch)
    target[index][pos] = 1
    index+=1
target = target[1:]

input_size = 24*5
hidden_size = 80
output_size = len(voice1Notes)
print(len(target), len(x_train))
#initialize -1 and 1
weights_hidden = np.random.randn(input_size, hidden_size)
biases_hidden = np.zeros(hidden_size)
weights_output = np.random.randn(hidden_size, output_size)
biases_output = np.zeros(output_size)


learning_rate = 0.01
num_epochs = 30
k = 5  # Number of folds
kf = KFold(n_splits=k, shuffle=False)

accuracy_scores = []

# Iterate over the folds
for train_index, test_index in kf.split(x_train):
    train_index = np.arange(train_index[0], train_index[-1]+1)
    test_index = np.arange(test_index[0], test_index[-1]+1)
    x_train_fold, x_test_fold = x_train[train_index], x_train[test_index]
    target_train_fold, target_test_fold = target[train_index], target[test_index]

    for epoch in range(num_epochs):
        # Forward propagation and backpropagation
        for i in range(len(x_train_fold)):
            y_hat, hidden_activations = forward_propagation([x_train_fold[i]])

            loss = -np.sum([target_train_fold[i]] * np.log(y_hat))

            output_error = y_hat - target_train_fold[i]
            grad_w2 = np.dot(hidden_activations.T, output_error)
            grad_b2 = np.sum(output_error, axis=0)

            hidden_error = np.dot(output_error, weights_output.T) * sigmoid_d(hidden_activations)
            grad_w1 = np.dot(np.array([x_train_fold[i]]).T, hidden_error)
            grad_b1 = np.sum(hidden_error, axis=0)

            weights_hidden -= learning_rate * grad_w1
            biases_hidden -= learning_rate * grad_b1
            weights_output -= learning_rate * grad_w2
            biases_output -= learning_rate * grad_b2

    # Evaluate the model on the testing fold
    y_hat_test = []
    for i in range(len(x_test_fold)):
        y_hat, _ = forward_propagation([x_test_fold[i]])
        y_hat_test.append(np.argmax(y_hat))

    accuracy = accuracy_score(np.argmax(target_test_fold, axis=1), y_hat_test)
    accuracy_scores.append(accuracy)

avg_accuracy = np.mean(accuracy_scores)
print("Average Accuracy:", avg_accuracy)

# Generate composition using the trained model
composition = []
previous_notes = x_train[len(x_train) - 1]
print(previous_notes)
for i in range(640):
    output, _ = forward_propagation([previous_notes])
    new_note = random.choices(voice1Notes, output[0])[0]
    composition.append(new_note)
    previous_notes = [previous_notes[5:]] + [changeToVector(new_note, voice1Notes)]
    previous_notes = [element for sublist in previous_notes for element in sublist]

print(composition)

csv_file_path = "/Users/amir0/Documents/NN/BACHSMASTERPIECE/output.csv"
# Open the CSV file in write mode
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Note"])
    for note in composition:
        writer.writerow([note])
