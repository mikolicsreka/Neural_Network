using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork
{

    public interface ISupervisedLearning
    {

        double Run(double[] input, double[] output, double[][] inp, double[][] outp);

        double RunEpoch(double[][] input, double[][] output);
    }
}
