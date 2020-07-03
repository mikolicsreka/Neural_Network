using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BackPropagation_Neural_Network
{

    public class Layer
    {
        public int NumberOfInputs { get; set; }
        public int NumberOfOutputs { get; set; }
        public int NumberOfNeurons { get; set; }
        public double[] Inputs { get; set; }
        public double[] Outputs { get; set; }
        public double[] Delta { get; set; }
        public double[,] WeightsDelta { get; set; }
        public double[,] Weights { get; set; }


        public Layer(int numberOfInputs, int numberOfOutputs)
        {
            //initilize datastructures
            Outputs = new double[numberOfInputs];
            Inputs = new double[numberOfInputs];
            Weights = new double[numberOfInputs, numberOfOutputs];
            Delta = new double[numberOfInputs];
            WeightsDelta = new double[numberOfInputs, numberOfOutputs];
        }
    }

    public class NN
    {
        double[] input;
        double[] target;
        double[] error;
        double avgError = 1000;
        readonly double eta;
        Layer hidden;
        Layer output;
       
       
        public double Phi(double input)
        {
            return (1 / (1 + Math.Pow(Math.E, -input)));
        }
        public double GetAvgError()
        {
            return avgError;
        }
        public void InitWeigths()
        {
            //init default weights
            Random rnd = new Random();
            for (int i = 0; i < hidden.Weights.GetLength(0); i++)
            {
                for (int j = 0; j < hidden.Weights.GetLength(1); j++)
                {
                    hidden.Weights[i, j] = rnd.NextDouble() - 0.5;
                }
            }
            for (int i = 0; i < output.Weights.GetLength(0); i++)
            {
                for (int j = 0; j < output.Weights.GetLength(1); j++)
                {
                    output.Weights[i, j] = rnd.NextDouble() - 0.5;
                }
            }
        }
        public double DerivativePhi(double input)
        {
            return Math.Pow(Math.E, -input) / Math.Pow(1 + Math.Pow(Math.E, -input), 2);
        }
        public double[] FeedForward(double[] input)
        {
            //feedforward
            //--input -> hidden
            this.input = input;
            for (int j = 0; j < hidden.Inputs.Length; j++)
            {
                hidden.Inputs[j] = 0;
                for (int i = 0; i < input.Length; i++)
                    hidden.Inputs[j] += hidden.Weights[j, i] * input[i];
            }
            for (int i = 0; i < hidden.Outputs.Length; i++)
                hidden.Outputs[i] = Phi(hidden.Inputs[i]);

            //--hidden ->output
            for (int j = 0; j < output.Inputs.Length; j++)
            {
                output.Inputs[j] = 0;
                for (int i = 0; i < hidden.Outputs.Length; i++)
                    output.Inputs[j] += output.Weights[j, i] * hidden.Outputs[i];
            }
            for (int i = 0; i < output.Outputs.Length; i++)
                output.Outputs[i] = Phi(output.Inputs[i]);

            return output.Outputs;
        }
        public void CalcError()
        {
            //--calc error
            for (int i = 0; i < target.Length; i++)
                error[i] = ((0.5) * (Math.Pow((target[i] - output.Outputs[i]), 2)));

            avgError = 0;
            for (int i = 0; i < error.Length; i++)
                avgError += (1 / input.Length) * error[i];
        }
        public void BackPropagation(double[] target)
        {
            //-backpropagation
            // k=1
            this.target = target;
            for (int l = 0; l < target.Length; l++)
                output.Delta[l] = (target[l] - output.Outputs[l]) * DerivativePhi(output.Inputs[l]);

            for (int j = 0; j < output.WeightsDelta.GetLength(0); j++)
                for (int i = 0; i < output.WeightsDelta.GetLength(1); i++)
                    output.WeightsDelta[j, i] = eta * output.Delta[j] * hidden.Outputs[i];

            //k=0
            for (int j = 0; j < hidden.Delta.Length; j++)
            {
                hidden.Delta[j] = 0;
                for (int k = 0; k < target.Length; k++)
                    hidden.Delta[j] += output.Delta[k] * output.Weights[k, j];

                hidden.Delta[j] *= DerivativePhi(hidden.Inputs[j]);
            }
            for (int j = 0; j < hidden.Inputs.Length; j++)
                for (int i = 0; i < input.Length; i++)
                    hidden.WeightsDelta[j, i] = eta * hidden.Delta[j] * input[i];

            CalcError();
            UpdateWeights();

        }
        public void UpdateWeights()
        {
            for (int j = 0; j < hidden.Weights.GetLength(0); j++)
                for (int i = 0; i < hidden.Weights.GetLength(1); i++)
                    hidden.Weights[j, i] += hidden.WeightsDelta[j, i];

            for (int j = 0; j < output.Weights.GetLength(0); j++)
                for (int i = 0; i < output.Weights.GetLength(1); i++)
                    output.Weights[j, i] += output.WeightsDelta[j, i];
        }
        public NN(int input, int target, int numberOfHiddenLayerPerceptron, double eta)
        {
            hidden = new Layer(numberOfHiddenLayerPerceptron, input);
            output = new Layer(target, numberOfHiddenLayerPerceptron);
            error = new double[target];
            InitWeigths();
            this.eta = eta;
        }
    }
}
