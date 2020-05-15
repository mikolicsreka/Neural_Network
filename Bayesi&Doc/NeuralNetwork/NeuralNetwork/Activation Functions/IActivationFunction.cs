using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Activation_Functions
{

    public interface IActivationFunction
    {

        double Function(double x);

        double Derivative(double x);

        double Derivative2(double y);
    }
}
