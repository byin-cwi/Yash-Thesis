import xml.etree.ElementTree as ET
import numpy as np
import keras
from keras.models import load_model

source = "Keras_model.h5"
target = "Pybrain_model.xml"
reference = "saved_net_24_16.xml"     # (4,24,16,2) configuration pybrain existing model


model = load_model(source)
layers = 3
w = []
b = []
for layer in range(layers): # code to print the weights and biases of each layer orderly
    weights = np.array(model.layers[layer].get_weights()[0])
    weights = np.round(np.float64(weights),8)
    weights = np.transpose(weights)
    weights = weights.reshape(-1)
    w.append(weights)

    biases = np.array(model.layers[layer].get_weights()[1])
    biases = np.round(np.float64(biases),8)
    biases = np.transpose(biases)
    biases = biases.reshape(-1)
    b.append(biases)
    print('Layer ',layer+1,' weights ',np.shape(w[layer]),':\n', w[layer].tolist())
    print('Layer ',layer+1,' biases ',np.shape(b[layer]),':\n', b[layer].tolist())

tree = ET.parse(reference)
root = tree.getroot()

# This loop pastes the .h5 file weights orderly into the .xml file at correct places.
for child in root.findall('Network'):      #Network
    for subchild in child.findall('Connections'):  # Connections
        for subsubchild in subchild.findall('FullConnection'):  # FullConnection
            for subnode in subsubchild.findall('inmod'):  # FullConnection    
                if subnode.get('val') == 'bias1':
                    # print(subnode.get('val'))
                    for node in subsubchild.findall('Parameters'):
                        # print(node.text)
                        node.text = str(b[0].tolist())     # Replace
                        # print(node.text)
                elif subnode.get('val') == 'bias2':
                    # print(subnode.get('val'))
                    for node in subsubchild.findall('Parameters'):
                        # print(node.text)
                        node.text = str(b[1].tolist())     # Replace
                        # print(node.text)
                elif subnode.get('val') == 'bias3':
                    # print(subnode.get('val'))
                    for node in subsubchild.findall('Parameters'):
                        # print(node.text)
                        node.text = str(b[2].tolist())     # Replace
                        # print(node.text)
                elif subnode.get('val') == 'in':
                    # print(subnode.get('val'))
                    for node in subsubchild.findall('Parameters'):
                        # print(node.text)
                        node.text = str(w[0].tolist())     # Replace
                        # print(node.text)
                elif subnode.get('val') == 'hidden1':
                    # print(subnode.get('val'))
                    for node in subsubchild.findall('Parameters'):
                        # print(node.text)
                        node.text = str(w[1].tolist())     # Replace
                        # print(node.text)
                elif subnode.get('val') == 'hidden2':
                    # print(subnode.get('val'))
                    for node in subsubchild.findall('Parameters'):
                        # print(node.text)
                        node.text = str(w[2].tolist())     # Replace
                        # print(node.text)
tree.write(target)        # store weights into this .xml file 
print('\033[92m'+'Successfully written to '+str(target)+" from "+str(source)+'\033[0m')


