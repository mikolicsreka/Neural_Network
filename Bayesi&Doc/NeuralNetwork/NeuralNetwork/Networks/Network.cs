using NeuralNetwork.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Networks
{
    public abstract class Network
    {
        protected int inputsCount;
        protected int layersCount;

        protected Layer[] layers;
        protected double[] output;

        public int InputsCount
        {
            get { return inputsCount; }
        }
        public int LayersCount
        {
            get { return layersCount; }
        }

        public double[] Output
        {
            get { return output; }
        }

        public Layer this[int index]
        {
            get { return layers[index]; }
        }


        protected Network(int inputsCount, int layersCount)
        {
            this.inputsCount = Math.Max(1, inputsCount);
            this.layersCount = Math.Max(1, layersCount);
            // create collection of layers
            layers = new Layer[this.layersCount];
        }

        public virtual double[] Compute(double[] input)
        {
            output = input;

            // compute each layer
            foreach (Layer layer in layers)
            {
                output = layer.Compute(output);
            }

            return output;
        }


        public virtual void Randomize()
        {
            foreach (Layer layer in layers)
            {
                layer.Randomize();
            }
        }
    }
}
