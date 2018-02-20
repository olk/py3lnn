#!/usr/bin/env python

# based on 'Neuronale Netze selbst programmieren', Tariq Rashid

import argparse
import nn
import numpy

# Instanz des Neuronalen Netzes erzeugen
def create( input_nodes, hidden_nodes, output_nodes, learning_rate):
    return nn.neural_network( input_nodes, hidden_nodes, output_nodes, learning_rate)

# Neuronales Netz trainieren
def train( n, training_data_path, output_nodes, epochs):
    # MNIST Trainingsdaten (CSV) laden
    with open( training_data_path, 'r') as training_data_file:
        training_data_lst = training_data_file.readlines()
    # Neuronales Netzwerk trainieren
    for e in range( epochs):
        # Iteration über alle Trainingsdaten
        for record in training_data_lst:
            # CVS aufsplitten
            all_values = record.split(',')
            # Input skalieren und Wertebereich auf (0,1) abbilden
            inputs = ( numpy.asfarray( all_values[1:])/255.0 * 0.99) + 0.01
            # außer dem Zielwert werden alle Tragetwerte werden auf 0.01 gesetzt
            targets = numpy.zeros( output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            n.train( inputs, targets)
    return

# Neuronales Netz testen
def test( n, test_data_path):
    # MNIST Testdaten (CSV) laden
    with open( test_data_path, 'r') as test_data_file:
        test_data_lst = test_data_file.readlines()
    # Neuronales Netzwerk testen
    scorecard = []
    # alles Datensätze iterieren
    for record in test_data_lst:
        # CSV ausplitten
        all_values = record.split(',')
        # erster Wert entsprcht der korrekten Antwort 
        correct_label = int( all_values[0])
        # Input skalieren und Wertebereich auf (0,1) abbilden
        inputs = ( numpy.asfarray( all_values[1:])/255.0 * 0.99) + 0.01
        # Netzwerk abfragen
        outputs = n.query( inputs)
        # der Index des höchten Wertes entspricht dem Label
        label = numpy.argmax( outputs)
        # Antwort speichern
        if ( label == correct_label):
            scorecard.append( 1)
        else:
            scorecard.append( 0)
    # Berechnung des Performance Scores
    scorecard_array = numpy.asarray( scorecard)
    print( "perfomance = ", scorecard_array.sum() / scorecard_array.size)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neuronales Netzwerk mit 3 Schichten.')
    parser.add_argument('--hidden-nodes', nargs='?', type=int, default=200)
    parser.add_argument('--training-data', nargs='?', type=str, default="mnist_dataset/mnist_train_100.csv")
    parser.add_argument('--test-data', nargs='?', type=str, default="mnist_dataset/mnist_test_10.csv")
    parser.add_argument('--learning-rate', nargs='?', type=float, default=0.1)
    parser.add_argument('--epochs', nargs='?', type=int, default=5)
    args = parser.parse_args()
    # Anzahl der Input- und Output-Nodes
    input_nodes = 784 # MNIST-Bild besteht aus 28x28 Pixel
    output_nodes = 10 # Zahlen von 0...9
    # Neuronales Netz erzeugen
    n = create( input_nodes, args.hidden_nodes, output_nodes, args.learning_rate)
    # Neuronales Netz trainieren
    train( n, args.training_data, output_nodes, args.epochs)
    # Neuronales Netz testen
    test( n, args.test_data)
