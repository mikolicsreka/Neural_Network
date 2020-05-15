using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Core
{
    using System;
    //A double típushoz egy range osztály min és max értékkel
    public class DoubleRange
    {
        private double min, max;

        public double Min
        {
            get { return min; }
            set { min = value; }
        }

        public double Max
        {
            get { return max; }
            set { max = value; }
        }

        //A range hossza
        public double Length
        {
            get { return max - min; }
        }

        //konstruktor
        public DoubleRange(double min, double max)
        {
            this.min = min;
            this.max = max;
        }

        //benne van e?
        public bool IsInside(double x)
        {
            return ((x >= min) && (x <= max));
        }

        public bool IsOverlapping(DoubleRange range)
        {
            return ((IsInside(range.min)) || (IsInside(range.max)));
        }
    }
}
