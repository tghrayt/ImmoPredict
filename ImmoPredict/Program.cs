using ImmoPredict.Models;

var predictor = new Predictor();

var input = new HouseData
{
    Surface = 75,
    Rooms = 3,
    LocationScore = 8
};

var price = predictor.Predict(input);

Console.WriteLine($"Prix estimé : {price}");