using Microsoft.ML;
using System.IO;
using System.Linq;

namespace MeerkatModel
{
    public class Classifier
    {
        private readonly MLContext _mLContext = new MLContext();
        private readonly PredictionEngine<ModelInput, ModelOutput> _predictionEngine;

        public Classifier()
        {
            var loadedModel = _mLContext.Model.Load("model\\trainedModel.zip", out _);
            _predictionEngine = _mLContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(loadedModel);
        }

        public string Classify(string imagePath)
        {
            var preProcessingPipeline = _mLContext
                .Transforms.LoadRawImageBytes(
                outputColumnName: "ImageBytes",
                imageFolder: Path.GetDirectoryName(imagePath),
                inputColumnName: "ImagePath");

            var imagePathAsArray = new[]
            {
                new
                {
                    ImagePath = imagePath,
                    Label = string.Empty
                }
            };

            var imagePathDataView = _mLContext.Data.LoadFromEnumerable(imagePathAsArray);

            var imageBytesDataview = preProcessingPipeline.Fit(imagePathDataView)
                .Transform(imagePathDataView);

            var modelInput = _mLContext.Data.CreateEnumerable<ModelInput>(
                imageBytesDataview,
                true)
                .First();

            ModelOutput prediction = _predictionEngine.Predict(modelInput);
            return prediction.PredictedLabel;
        }
    }
}