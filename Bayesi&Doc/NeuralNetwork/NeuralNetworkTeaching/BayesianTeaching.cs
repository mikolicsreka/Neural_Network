using NeuralNetwork.Activation_Functions;
using NeuralNetwork.Learning;
using NeuralNetwork.Networks;
using System;
using System.Globalization;
using System.IO;

namespace NeuralNetworkTeaching
{
    class BayesianTeaching
    {
        public BayesianTeaching()
        {
            LoadData();
            LeaningSetup();
            Learn();
        }
        private double[,] data = null;
        private double[][] testingData = new double[30][];

        private double learningRate = 0.1;
        private double momentum = 0.0;
        private double sigmoidAlphaValue = 2.0;
        private int neuronsInFirstLayer = 20;
        private int iterations = 100;

        private bool needToStop = false;

        private void LoadData()
        {
            StreamReader reader = null;
            // read maximum 50 points
            double[,] tempData = new double[200, 2];
            double minX = double.MaxValue;
            double maxX = double.MinValue;
            try
            {
                reader = File.OpenText("file4.txt");
                string str = null;
                int i = 0;

                // read the data
                while ((i < 200) && ((str = reader.ReadLine()) != null))
                {
                    string[] strs = str.Split(' ');

                    tempData[i, 0] = double.Parse(strs[0], CultureInfo.InvariantCulture);
                    tempData[i, 1] = double.Parse(strs[1], CultureInfo.InvariantCulture);

                    // search for min value
                    if (tempData[i, 0] < minX)
                        minX = tempData[i, 0];
                    // search for max value
                    if (tempData[i, 0] > maxX)
                        maxX = tempData[i, 0];

                    i++;
                }

                // allocate and set data
                data = new double[i, 2];
                Array.Copy(tempData, 0, data, 0, i * 2);
            }
            catch (Exception)
            {
                Console.WriteLine("An error has occured while reading the file");
                return;
            }
            finally
            {
                // close file
                if (reader != null)
                    reader.Close();
            }


        }

        private void LeaningSetup()
        {
            learningRate = 0.1;
            // sigmoid's alpha value
            sigmoidAlphaValue = 2;
            // get neurons count in first layer
            neuronsInFirstLayer = 10;
            // iterations
            iterations = 1000;
        }

        private void Learn()
        {
            int samples = data.GetLength(0);
            double[][] input = new double[samples][];
            double[][] output = new double[samples][];

            for (int i = 0; i < samples; i++)
            {
                input[i] = new double[1];
                output[i] = new double[1];
                input[i][0] = (data[i, 0]);
                output[i][0] = (data[i, 1]);
            }

            ActivationNetwork network = new ActivationNetwork(
                new BipolarSigmoidFunction(sigmoidAlphaValue),
                1, neuronsInFirstLayer, 1);
            BayesianLearning teacher = new BayesianLearning(network);
            teacher.LearningRate = learningRate;

            int iteration = 1;
            while (!needToStop)
            {
                double error = teacher.RunEpoch(input, output) / samples;
                Console.WriteLine("The avg error is: " + error);
                iteration++;

                //Mikor álljunk meg? Ha az iterációk száma eléri a megadottat, vagy a hiba kisebb mint valami
                if ((iterations != 0) && (iteration > iterations) || error < 1)
                {
                    for (int i = 0, n = input.Length; i < n; i++)
                    {
                        WriteData(input[i], teacher);
                    }
                    break;

                }
            }
           
            LoadTestData();



            for (int i = 0; i < 30; i++)
            {
                double[] test = teacher.network.Compute(testingData[i]);

            } 



        }

        public void WriteData(double[] input, BayesianLearning teacher)
        {
            double[] writeOutput = teacher.network.Compute(input);
            using (StreamWriter writetext = new StreamWriter("log.txt", append: true))
            {
                for (int i = 0; i < writeOutput.Length; i++)
                {
                    writetext.WriteLine(input[i] + " " + writeOutput[i]);
                }
                
            }

        }

        private void LoadTestData()
        {
            StreamReader reader = null;
            // read maximum 50 points
            double[,] testData = new double[30, 1];
            try
            {
                // open selected file
                reader = File.OpenText("test.txt");
                string str = null;

                int i = 0;
                // read the data
                while ((i < 30) && ((str = reader.ReadLine()) != null))
                {
                    testData[i, 0] = double.Parse(str, CultureInfo.InvariantCulture);
                    ++i;
                }

                for (int j = 0; j < 30; j++)
                {
                    testingData[j] = new double[1];

                    testingData[j][0] = (testData[j, 0]);

                }


            }
            catch (Exception ex)
            {
                Console.WriteLine(ex);
            }



        }
        public static void Main(string[] args)
        {

            BayesianTeaching teach = new BayesianTeaching();
        }

    }
}














