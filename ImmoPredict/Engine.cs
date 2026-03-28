using ImmoPredict.Models;
using Microsoft.ML;

namespace ImmoPredict;

public class Engine
{
    private readonly MLContext _mlContext = new MLContext();
    private readonly string _dataPath = Path.Combine(
        Path.GetDirectoryName(typeof(Engine).Assembly.Location),
        "DataSet",
        "data.csv");
    private const string inputColumnName = "Features";

    public void Treatment()
    {
        Console.WriteLine($"Début de traitement");
        Console.WriteLine("Loading data...");
        var data = _mlContext.Data.LoadFromTextFile<HouseData>(
            path: _dataPath,
            hasHeader: true,
            separatorChar: ',');

        var split = _mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

        var pipeline = _mlContext.Transforms
            .Concatenate(
                inputColumnName,
                nameof(HouseData.Surface),
                nameof(HouseData.Rooms),
                nameof(HouseData.LocationScore))
            .Append(_mlContext.Regression.Trainers.Sdca(
                labelColumnName: nameof(HouseData.Price),
                featureColumnName: inputColumnName));

        Console.WriteLine("Training model...");
        var model = pipeline.Fit(split.TrainSet);

        Console.WriteLine("Evaluating model...");
        var predictions = model.Transform(split.TestSet);

        Console.WriteLine("Calculating metrics...");
        var metrics = _mlContext.Regression.Evaluate(
            predictions,
            labelColumnName: nameof(HouseData.Price));

        Console.WriteLine($"R² : {metrics.RSquared}");
        Console.WriteLine($"RMSE : {metrics.RootMeanSquaredError}");

        _mlContext.Model.Save(model, data.Schema, "model.zip");
    }
}
