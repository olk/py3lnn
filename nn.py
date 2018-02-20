# based on 'Neuronale Netze selbst programmieren', Tariq Rashid

import numpy
import scipy.special

class neural_network:
    # Initialisierung
    def __init__( self, inodes, hnodes, onodes, lr):
        # Anzahl der Knoten der Input-, Hidden- und Output-Schicht
        self.inodes = inodes
        self.hnodes = hnodes
        self.onodes = onodes
        # Initialisierung der Gwewichts-Matrizen mit zufälligen Werten aus (0,1)
        # wih: Matrix für Übergang von Input -> Hidden
        # who: Matrix für Übergang von Hidden -> Output
        self.wih = numpy.random.normal( 0.0, pow( self.hnodes, -0.5), (self.hnodes, self.inodes) )
        self.who = numpy.random.normal( 0.0, pow( self.onodes, -0.5), (self.onodes, self.hnodes) )
        # Lernrate
        self.lr = lr
        # Sigmoid == Aktivierungsfunktion 
        self.activation_fn = lambda x: scipy.special.expit( x)
        return

    # Training
    def train( self, input_lst, target_lst):
        # Konvertierung der Inputlisten zu 2D-Arrays
        inputs = numpy.array( input_lst, ndmin = 2).T
        targets = numpy.array( target_lst, ndmin = 2).T
        # in Hidden Layer eingehende Signale
        input_hidden = numpy.dot( self.wih, inputs)
        # aus Hidden Layer austretende Signale
        output_hidden = self.activation_fn( input_hidden)
        # in Output Layer eingehende Signale
        input_final = numpy.dot( self.who, output_hidden)
        # aus Output Layer austretende Signale
        output_final = self.activation_fn( input_final)
        # Fehler im Output Layer entspricht der Differenz (targets - inputs)
        error_output = targets - output_final
        # Fehler des Hidden Layer entspricht dem error_output, gesplittet
        # nach entsprechend den Geweichten, rekombiniert an den hidden nodes
        error_hidden = numpy.dot( self.who.T, error_output)
        # Änderung der Gewichte bzgl. der Verbindungen zw. Hidden and Output Layer
        self.who += self.lr * numpy.dot( (error_output * output_final * (1.0 - output_final)), numpy.transpose( output_hidden) )
        # Änderung der Gewichte bzgl. der Verbindungen zw. Input and Hidden Layer
        self.wih += self.lr * numpy.dot( (error_hidden * output_hidden * (1.0 - output_hidden)), numpy.transpose( inputs) )
        return

    # Abfrage
    def query( self, input_lst):
        # Konvertierung der Inputlisten zu 2D-Arrays
        inputs = numpy.array( input_lst, ndmin = 2).T
        # in Hidden Layer eingehende Signale
        input_hidden = numpy.dot( self.wih, inputs)
        # aus Hidden Layer austretende Signale
        output_hidden = self.activation_fn( input_hidden)
        # in Output Layer eingehende Signale
        input_final = numpy.dot( self.who, output_hidden)
        # aus Output Layer austretende Signale
        output_final = self.activation_fn( input_final)
        return output_final
