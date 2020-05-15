using NeuralNetwork.Activation_Functions;
using NeuralNetwork.Neurons;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Layers
{

    public class ActivationLayer : Layer
    {

        public new ActivationNeuron this[int index]
        {
            get { return (ActivationNeuron)neurons[index]; }
        }

        public ActivationLayer(int neuronsCount, int inputsCount, IActivationFunction function)
                            : base(neuronsCount, inputsCount)
        {
            for (int i = 0; i < neuronsCount; i++)
                neurons[i] = new ActivationNeuron(inputsCount, function);
        }

    }
}
