using NeuralNetwork.Activation_Functions;
using NeuralNetwork.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Networks
{

    public class ActivationNetwork : Network
    {
 
        public new ActivationLayer this[int index]
        {
            get { return ((ActivationLayer)layers[index]); }
        }
        public ActivationNetwork(IActivationFunction function, int inputsCount, params int[] neuronsCount)
                            : base(inputsCount, neuronsCount.Length)
        {
            // create each layer
            for (int i = 0; i < layersCount; i++)
            {
                layers[i] = new ActivationLayer(
                    // neurons count in the layer
                    neuronsCount[i],
                    // inputs count of the layer
                    (i == 0) ? inputsCount : neuronsCount[i - 1],
                    // activation function of the layer
                    function);
            }
        }
    }
}
