using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;

namespace BackPropagation_Neural_Network
{
    class Program
    {
        static void Main(string[] args)
        {
            Config cfg = new Config("simulparams.txt");

            DateTime start = DateTime.Now;
            System.IO.StreamWriter file = new System.IO.StreamWriter("../../docs/outputs/"+cfg.output_file);
            NN NeuralN = new NN(1, 1, cfg.hidden, cfg.eta);
            List<Values> data = GetData(cfg.input_file);
            
            while (NeuralN.GetAvgError() > cfg.tol)
            {
                for (int i = 0; i < cfg.teach; i++)
                {
                    NeuralN.FeedForward(new double[] { data[i].n1 });
                    NeuralN.BackPropagation(new double[] { data[i].n2 });
                }
                Console.Write(".");
            }
            Console.WriteLine();
            Console.WriteLine("{0,20} {1,20} {2,20}", "Input: ", " Output: " , " Target: " );
            Process proc = Process.GetCurrentProcess();

            for (int i = cfg.teach; i < cfg.all; i++)
            {
                Console.WriteLine("{0,20:N14} {1,20:N14} {2,20:N14}", data[i].n1, NeuralN.FeedForward(new double[] { data[i].n1 })[0], data[i].n2);
                string line = data[i].n1 + " " + NeuralN.FeedForward(new double[] { data[i].n1 })[0];
                file.WriteLine(line);
            }
            file.Close();
            Console.WriteLine("Average error: " + NeuralN.GetAvgError());
            TimeSpan timeItTook = DateTime.Now - start;
            Console.WriteLine(timeItTook);

            Console.WriteLine(proc.PrivateMemorySize64/1024 + "Kbyte " );
            Console.ReadLine();

        }
        public static List<Values> GetData(string filename)
        {
            string path = "../../docs/inputs/" + filename;
            string[] lines = System.IO.File.ReadAllLines(path);
            List<Values> data = new List<Values>();
            for (int i = 0; i < lines.Length; i++)
            {
                try
                {
                    string[] splitted = lines[i].Split('#');
                    Values tmp = new Values(double.Parse(splitted[0]), double.Parse(splitted[1]));
                    data.Add(tmp);
                }
                catch (Exception e)
                {
                    Console.WriteLine(e.ToString());
                }
            }
            return data;
        }
        
    }
    class Values
    {
        public double n1 { get; set; }
        public double n2 { get; set; }
        public Values(double n1, double n2)
        {
            this.n1 = n1;
            this.n2 = n2;
        }

    }
    class Config
    {
        public int all { get; set; }
        public int teach { get; set; }
        public double tol { get; set; }
        public string input_file { get; set; }
        public string output_file { get; set; }
        public double eta { get; set; }
        public int hidden { get; set; }

        public Config(int all, int teach, double tol, string input_file, string output_file, double eta, int hidden)
        {
            this.all = all;
            this.teach = teach;
            this.tol = tol;
            this.input_file = input_file;
            this.output_file = output_file;
            this.eta = eta;
            this.hidden = hidden;
        }
        public Config(string filename)
        {
            string path = "../../docs/" + filename;
            string[] lines = System.IO.File.ReadAllLines(path);
            this.all = int.Parse(lines[0].Split(' ')[1]);
            this.teach = int.Parse(lines[1].Split(' ')[1]);
            this.tol = double.Parse(lines[2].Split(' ')[1]);
            this.input_file = lines[3].Split(' ')[1];
            this.output_file = lines[4].Split(' ')[1];
            this.eta = double.Parse(lines[5].Split(' ')[1]);
            this.hidden = int.Parse(lines[6].Split(' ')[1]);
        }
    }
    

}
