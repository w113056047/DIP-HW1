
using OpenCvSharp;

double[,] pair1 = { { 134, 294 }, { 619, 294 } };
double[,] pair2 = { { 174, 499 }, { 676, 472 } };
double[,] pair3 = { { 134, 716 }, { 659, 650 } };

double[,] points = { { pair1[0, 0], pair1[0, 1], 1 }, { pair2[0, 0], pair2[0, 1], 1 }, { pair3[0, 0], pair3[0, 1], 1 } };

double[] xVals = [pair1[1, 0], pair2[1, 0], pair3[1, 0]];
double[] yVals = [pair1[1, 1], pair2[1, 1], pair3[1, 1]];

var xDst = new List<double>();
var yDst = new List<double>();

Cv2.Solve(
    InputArray.Create(points), InputArray.Create(xVals),
    OutputArray.Create(xDst),
    DecompTypes.LU);

Cv2.Solve(
    InputArray.Create(points), InputArray.Create(yVals),
    OutputArray.Create(yDst),
    DecompTypes.LU);

Console.WriteLine("a: {0}, b: {1}, c: {2}, d: {3}, e: {4}, f: {5}", xDst[0], xDst[1], xDst[2], yDst[0], yDst[1], yDst[2]);

using var origin = new Mat("c1.png", ImreadModes.Color);
using var inputImage = new Mat("c2.png", ImreadModes.Color);
var inputImageMat = new Mat<Vec3b>(inputImage);
var inputImageIndexer = inputImageMat.GetIndexer();

var output = new Mat<Vec3b>(origin.Rows + inputImage.Rows, origin.Cols + inputImage.Cols);

var a = xDst[0];
var b = xDst[1];
var c = xDst[2];
var d = yDst[0];
var e = yDst[1];
var f = yDst[2];

for (int y = 0; y < inputImage.Height; y++)
{
    for (int x = 0; x < inputImage.Width; x++)
    {
        Vec3b color = inputImageIndexer[y, x];
        var newY = (int)(a * y + b * x + c);
        var newX = (int)(d * y + e * x + f);
        output.Set(newY, newX, color);
    }
}

Cv2.ImShow("output", output);
Cv2.WaitKey();
Cv2.DestroyAllWindows();