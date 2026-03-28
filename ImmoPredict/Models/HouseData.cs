using Microsoft.ML.Data;


namespace ImmoPredict.Models;
public class HouseData
{
    [LoadColumn(0)]
    public float Surface { get; set; }

    [LoadColumn(1)]
    public float Rooms { get; set; }

    [LoadColumn(2)]
    public float LocationScore { get; set; }

    [LoadColumn(3)]
    public float Price { get; set; }
}

public class HousePrediction
{
    public float Score { get; set; }
}


