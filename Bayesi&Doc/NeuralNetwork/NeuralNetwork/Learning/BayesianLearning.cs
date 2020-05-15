using NeuralNetwork.Activation_Functions;
using NeuralNetwork.Layers;
using NeuralNetwork.Networks;
using NeuralNetwork.Neurons;
using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Text;

namespace NeuralNetwork.Learning
{

    public class BayesianLearning : ISupervisedLearning
    {
        double[,] neuronChanges;

        int counting = 0;
        //tanítandó network
        public ActivationNetwork network;
        // learning rate
        private double learningRate = 0.1;

        // neuron's errors
        private double[][] neuronErrors = null;
        // weight's updates
        private double[][][] weightsUpdates = null;
        // threshold's updates
        private double[][] thresholdsUpdates = null;

        private double previousError = 10000;

        ActivationNeuron randomNeuronG;
        int randomEdgeG;
        double randomWeighUpdateG;

        ///Tanulás gyorsasága. Default: 0.1
        public double LearningRate
        {
            get { return learningRate; }
            set
            {
                learningRate = Math.Max(0.0, Math.Min(1.0, value));
            }
        }

        public BayesianLearning(ActivationNetwork network)
        {
           
            this.network = network;

            // create error and deltas arrays
            neuronErrors = new double[network.LayersCount][];
            weightsUpdates = new double[network.LayersCount][][];
            thresholdsUpdates = new double[network.LayersCount][];

            // initialize errors and deltas arrays for each layer
            for (int i = 0, n = network.LayersCount; i < n; i++)
            {
                Layer layer = network[i];

                neuronErrors[i] = new double[layer.NeuronsCount];
                weightsUpdates[i] = new double[layer.NeuronsCount][];
                thresholdsUpdates[i] = new double[layer.NeuronsCount];

                // for each neuron
                for (int j = 0; j < layer.NeuronsCount; j++)
                {
                    weightsUpdates[i][j] = new double[layer.InputsCount];
                }
            }
        }

        double error;
        public double Run(double[] input, double[] output, double[][] inp, double[][] outp)
        {
            // calculate weights updates
            while (true)
            {
                CalculateUpdates(input);

                network.Compute(input);
                 //divide by input num
                double avgerror = AvgError(inp, outp) / 200;
                error = CalculateError(output);
                if (avgerror < previousError || avgerror < 0.03)
                {
                    Console.WriteLine("AVG ERRROR: " + avgerror);
                    previousError = avgerror;
                    break;
                }
                else
                {
                    deleteChanges();
                }
            }
            return error;
        }
        public double RunEpoch(double[][] input, double[][] output)
        {
            double error = 0.0;

            // run learning procedure for all samples
            for (int i = 0, n = input.Length; i < n; i++)
            {
                error += Run(input[i], output[i], input, output);
            }

            // return summary error
            return error;
        }


        private double CalculateError(double[] desiredOutput)
        {

            ActivationLayer layer;

            double error = 0, e;

            double output;

            int layersCount = network.LayersCount;

            layer = network[layersCount - 1];

            for (int i = 0; i < desiredOutput.Length; i++)
            {
                output = layer[i].Output;
                e = desiredOutput[i] - output;
                error += (e * e);

                counting += 1;
            }
            return error / 2.0;
        }

        public double AvgError(double[][] input, double[][] output)
        {
            double error = 0.0;

            for (int i = 0, n = input.Length; i < n; i++)
            {
                error += Error(input[i], output[i]);
            }

            return error;
        }

        public double Error(double[] input, double[] output)
        {
            network.Compute(input);
            error = CalculateError(output);

            return error;

        }


        public double GetRandomNumber(double minimum, double maximum)
        {
            Random random = new Random();
            return random.NextDouble() * (maximum - minimum) + minimum;
        }

        private void CalculateUpdates(double[] input)
        {
            ActivationNeuron neuron;
            ActivationNeuron[] edge;

            ActivationLayer layer;
            
            neuronChanges = new double[network.LayersCount,100];
            for (int i = 0, n = network.LayersCount; i < n; i++)
            {
                layer = network[i];

                for (int j = 0, m = layer.NeuronsCount; j < m; j++)
                {
                    neuron = layer[j];

                    for (int k = 0, s = neuron.InputsCount; k < s; k++)
                    {

                        double randomWeight = GetRandomNumber(-0.5, 0.5);
                        neuronChanges[i,k] = randomWeight;
                        neuron[k] += randomWeight;
                    }
                }
            }

        }

        private void CalculateAgain(ActivationNeuron neuron, int edge)
        {
            double randomWeight = GetRandomNumber(-0.5, 0.5);
            randomWeighUpdateG = randomWeight;
            neuron[edge] += randomWeight;
        }

        private void deleteChanges()
        {
            ActivationNeuron neuron;
            ActivationNeuron[] edge;
            ActivationLayer layer;

            for (int i = 0, n = network.LayersCount; i < n; i++)
            {
                layer = network[i];

                for (int j = 0, m = layer.NeuronsCount; j < m; j++)
                {
                    neuron = layer[j];

                    for (int k = 0, s = neuron.InputsCount; k < s; k++)
                    {

                        neuron[k] -= neuronChanges[i,k];
                    }
                }
            }
        }

    }


}
