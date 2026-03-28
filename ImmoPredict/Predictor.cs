using ImmoPredict.Models;
using Microsoft.ML;

public class Predictor
{
    private readonly MLContext _mlContext = new MLContext();
    private readonly PredictionEngine<HouseData, HousePrediction> _engine;

    public Predictor()
    {
        var model = _mlContext.Model.Load("model.zip", out _);

        _engine = _mlContext.Model.CreatePredictionEngine<HouseData, HousePrediction>(model);
    }

    public float Predict(HouseData input)
    {
        var result = _engine.Predict(input);
        return result.Score;
    }
}