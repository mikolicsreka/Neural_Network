using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Neurons
{
    using System;
    using NeuralNetwork;
    using NeuralNetwork.Core;

    public abstract class Neuron
    {
        protected int inputsCount = 0;
        protected double[] weights = null;
        protected double output = 0;

        public int InputsCount
        {
            get { return inputsCount; }
        }

        public double Output
        {
            get { return output; }
        }

        //Neuron konstruktora
        public Neuron(int inputs)
        {
            //súlyok elérése
            inputsCount = Math.Max(1, inputs);
            weights = new double[inputsCount];
            //neuron randomizálása
            Randomize();
        }

        //Random generátor
        //A neuronok súlyának beálltásához
        protected static Random rand = new Random((int)DateTime.Now.Ticks);

        //Range a random generátornak, x-y közötti értékek legyenek
        protected static DoubleRange randRange = new DoubleRange(-0.5,0.5);

        //Random generátor
        //A neuronok súlyának beállításához
        public static Random RandGenerator
        {
            get { return rand; }
            set
            {
                if (value != null)
                {
                    rand = value;
                }
            }
        }

        //Random generátor range
        public static DoubleRange RandRange
        {
            get { return randRange; }
            set
            {
                if (value != null)
                {
                    randRange = value;
                }
            }
        }

        //Neuron súly elérése
        public double this[int index]
        {
            get { return weights[index]; }
            set { weights[index] = value; }
        }

        //Neuron randomizálása
        //A súlyokat random értékekkel inicializálja az adott rangen belül
        public virtual void Randomize()
        {
            double d = randRange.Length;

            // randomize weights
            for (int i = 0; i < inputsCount; i++)
                weights[i] = rand.NextDouble() * d + randRange.Min;
        }

        //Kiszámolja az outputját a neuronnak
        public abstract double Compute(double[] input);

    }
}
