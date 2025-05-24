using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.Globalization;
using CsvHelper;
using CsvHelper.Configuration;
using MathNet.Numerics.LinearAlgebra;
using NCalc;

public class WidgetParameters
{
    public int n { get; set; }
    public double beta { get; set; }
    public double sigma { get; set; }
    public double T { get; set; }
    public int num_points { get; set; }
    public double epsilon { get; set; }
    public string function { get; set; }
}

public class Results
{
    public double LaguerreFunctionResult { get; set; }
    public double CutoffT { get; set; }
    public double Integral { get; set; }
    public string Transform { get; set; }
    public double InverseTransform { get; set; }
}

public class Laguerre
{
    private double _beta;
    private double _sigma;

    public Laguerre(double beta = 2, double sigma = 4)
    {
        Beta = beta;
        Sigma = sigma;
    }

    public double Beta
    {
        get => _beta;
        set
        {
            if (value > 0)
                _beta = value;
            else
                throw new ArgumentException("Beta must be positive");
        }
    }

    public double Sigma
    {
        get => _sigma;
        set
        {
            if (value > 0)
                _sigma = value;
            else
                throw new ArgumentException("Sigma must be positive");
        }
    }

    public double LaguerreFunction(double t, int n)
    {
        double L_0 = Math.Sqrt(Sigma) * Math.Exp(-Beta / 2.0 * t);
        if (n == 0)
            return L_0;

        double L_1 = Math.Sqrt(Sigma) * (1 - Sigma * t) * Math.Exp(-Beta / 2.0 * t);
        if (n == 1)
            return L_1;

        double L_nm2 = L_0;
        double L_nm1 = L_1;
        double L_n = 0;
        for (int k = 2; k <= n; k++)
        {
            L_n = ((2 * k - 1 - Sigma * t) / k) * L_nm1 - ((k - 1.0) / k) * L_nm2;
            L_nm2 = L_nm1;
            L_nm1 = L_n;
        }

        return L_n;
    }

    public Dictionary<string, List<double>> TabulateLaguerre(int n, double T = 10, int numPoints = 100)
    {
        var tValues = MathNet.Numerics.LinearAlgebra.Vector<double>.Build.Dense(numPoints, i => i * T / (numPoints - 1)).ToList();
        var result = new Dictionary<string, List<double>>();

        for (int k = 0; k <= n; k++)
        {
            string key = $"l_{k}(t)";
            result[key] = tValues.Select(t => LaguerreFunction(t, k)).ToList();
        }
        result["t"] = tValues;

        return result;
    }

    public (double T, List<Dictionary<string, double>> Table) FindCutoffT(int N, double epsilon = 1e-3, double initialT = 1, double step = 0.1)
    {
        double T = initialT;
        while (true)
        {
            bool valid = true;
            var valuesTable = new List<Dictionary<string, double>>();

            for (int n = 0; n <= N; n++)
            {
                double lValue = Math.Abs(LaguerreFunction(T, n));
                valuesTable.Add(new Dictionary<string, double>
                {
                    {"n", n},
                    {"T", T},
                    {"|l_n(T)|", lValue}
                });

                if (lValue >= epsilon)
                    valid = false;
            }

            if (valid)
                return (T, valuesTable);

            T += step;
        }
    }

    public double LaguerreIntegralSimpson(Func<double, double> f, int k, double alpha, double T, double epsilon = 1e-6, int numPoints = 1000)
    {
        if (numPoints % 2 != 0)
            numPoints++;

        double h = T / numPoints;
        var tValues = MathNet.Numerics.LinearAlgebra.Vector<double>.Build.Dense(numPoints + 1, i => i * h).ToList();

        var integrand = tValues.Select(t => f(t) * LaguerreFunction(t, k) * Math.Exp(-alpha * t)).ToList();

        double integral = h / 3.0 * (
            integrand[0] +
            2 * integrand.Skip(2).Take(integrand.Count - 3).Where((_, i) => i % 2 == 0).Sum() +
            4 * integrand.Skip(1).Where((_, i) => i % 2 == 0).Sum() +
            integrand[^1]
        );

        return integral;
    }

    public double InverseLaguerreTransform(double[] hN, double t, double Beta)
    {
        double actualAlpha = Beta;
        return Enumerable.Range(0, hN.Length)
            .Sum(k => hN[k] * LaguerreFunction(t, k));
    }

    public double[] LaguerreTransform(Func<double, double> f, int N, double Beta, double T = 2 * Math.PI, double epsilon = 1e-6)
    {
        double actualAlpha = Beta;
        return Enumerable.Range(0, N + 1)
            .Select(k => LaguerreIntegralSimpson(f, k, actualAlpha, T, epsilon))
            .ToArray();
    }
}

class Program
{
    static void Main()
    {
        try
        {
            // 1. Зчитуємо параметри з CSV-файлу
            string inputCsvPath = "laguerre_params.csv";
            WidgetParameters parameters;

            using (var reader = new StreamReader(inputCsvPath))
            using (var csv = new CsvReader(reader, CultureInfo.InvariantCulture))
            {
                parameters = csv.GetRecords<WidgetParameters>().First();
            }

            var laguerre = new Laguerre(parameters.beta, parameters.sigma);

            // 2. Парсимо функцію з CSV за допомогою NCalc
            Func<double, double> f;
            try
            {
                var expression = new Expression(parameters.function);
                expression.Parameters["t"] = 0.0;
                f = t =>
                {
                    expression.Parameters["t"] = t;
                    return Convert.ToDouble(expression.Evaluate());
                };

                double testValue = f(0.0);
            }
            catch (Exception ex)
            {
                throw new ArgumentException($"Невірний формат функції '{parameters.function}': {ex.Message}");
            }

            Results results = new Results();

            results.LaguerreFunctionResult = laguerre.LaguerreFunction(4, parameters.n);
            Console.WriteLine($"Laguerre Function (t=4, n={parameters.n}): {results.LaguerreFunctionResult}");

            var tabulated = laguerre.TabulateLaguerre(parameters.n, parameters.T, parameters.num_points);

            var (cutoffT, cutoffTable) = laguerre.FindCutoffT(parameters.n, parameters.epsilon);
            results.CutoffT = cutoffT;
            Console.WriteLine($"Cutoff T: {results.CutoffT}");

            results.Integral = laguerre.LaguerreIntegralSimpson(f, k: 2, alpha: parameters.sigma, T: parameters.T, epsilon: parameters.epsilon);
            Console.WriteLine($"Integral (k=2, alpha={parameters.sigma}, T={parameters.T}): {results.Integral}");

            double[] transform = laguerre.LaguerreTransform(f, parameters.n, Beta: parameters.beta, T: parameters.T, epsilon: parameters.epsilon);
            results.Transform = string.Join(";", transform);
            Console.WriteLine($"Laguerre Transform: {results.Transform}");

            results.InverseTransform = laguerre.InverseLaguerreTransform(transform, t: 4.0, Beta: parameters.beta);
            Console.WriteLine($"Inverse Transform (t=4): {results.InverseTransform}");

            string outputCsvPath = $"laguerre_results.csv";
            using (var writer = new StreamWriter(outputCsvPath))
            using (var csv = new CsvWriter(writer, CultureInfo.InvariantCulture))
            {
                // Записуємо параметри
                csv.WriteRecord(parameters);
                csv.NextRecord();

                // Додаємо роздільник
                csv.WriteField("---");
                csv.NextRecord();

                // Записуємо результати
                csv.WriteRecord(results);
                csv.NextRecord();

                // Записуємо таблицю відсічення
                csv.WriteField("Cutoff Table");
                csv.NextRecord();
                csv.WriteField("n");
                csv.WriteField("T");
                csv.WriteField("|l_n(T)|");
                csv.NextRecord();

                foreach (var row in cutoffTable)
                {
                    csv.WriteField(row["n"]);
                    csv.WriteField(row["T"]);
                    csv.WriteField(row["|l_n(T)|"]);
                    csv.NextRecord();
                }
            }

            var config = new CsvConfiguration(CultureInfo.InvariantCulture)
            {
                Delimiter = "\t"
            };
            // Записуємо табуляцію в окремий файл tabulation_results.csv
            string tabulationCsvPath = "tabulation_results.csv";
            using (var writer = new StreamWriter(tabulationCsvPath))
            using (var csv = new CsvWriter(writer, config))
            {
                // Записуємо заголовки для табуляції
                var headers = tabulated.Keys.ToList();
                foreach (var header in headers)
                {
                    csv.WriteField(header);
                }
                csv.NextRecord();

                // Записуємо дані табуляції
                int rowCount = tabulated["t"].Count;
                for (int i = 0; i < rowCount; i++)
                {
                    foreach (var header in headers)
                    {
                        csv.WriteField(tabulated[header][i]);
                    }
                    csv.NextRecord();
                }
            }

            Console.WriteLine($"Результати збережено у файл: {outputCsvPath}");
            Console.WriteLine($"Табуляція збережена у файл: {tabulationCsvPath}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Помилка: {ex.Message}");
        }
    }
}