using OpenCvSharp;

// 讀取圖片
List<string> inputFiles = ["image1.jpeg", "image2.jpeg", "image3.jpeg"];

// x,y 的 padding
int yPadding = 100;
int xPadding = 100;

// 讀取錨點座標
var anchorPairs = File.ReadAllLines("coordinates.csv")
    .Select(line => line.Split(','))
    .Select(pair => new double[,] { { double.Parse(pair[0]), double.Parse(pair[1]) }, { double.Parse(pair[2]), double.Parse(pair[3]) } })
    .ToList();

double[,] pair1 = anchorPairs[0];
double[,] pair2 = anchorPairs[1];
double[,] pair3 = anchorPairs[2];
double[,] pair4 = anchorPairs[3];
double[,] pair5 = anchorPairs[4];
double[,] pair6 = anchorPairs[5];

Console.WriteLine($"Img1:");
Console.WriteLine($"\tpair1: ({pair1[0, 0]}, {pair1[0, 1]}) <-> ({pair1[1, 0]}, {pair1[1, 1]})");
Console.WriteLine($"\tpair2: ({pair2[0, 0]}, {pair2[0, 1]}) <-> ({pair2[1, 0]}, {pair2[1, 1]})");
Console.WriteLine($"\tpair3: ({pair3[0, 0]}, {pair3[0, 1]}) <-> ({pair3[1, 0]}, {pair3[1, 1]})");
Console.WriteLine($"Img2:");
Console.WriteLine($"\tpair4: ({pair4[0, 0]}, {pair4[0, 1]}) <-> ({pair4[1, 0]}, {pair4[1, 1]})");
Console.WriteLine($"\tpair5: ({pair5[0, 0]}, {pair5[0, 1]}) <-> ({pair5[1, 0]}, {pair5[1, 1]})");
Console.WriteLine($"\tpair6: ({pair6[0, 0]}, {pair6[0, 1]}) <-> ({pair6[1, 0]}, {pair6[1, 1]})");

double[,] points1 = { { pair1[0, 0], pair1[0, 1], 1 }, { pair2[0, 0], pair2[0, 1], 1 }, { pair3[0, 0], pair3[0, 1], 1 } };
double[,] points2 = { { pair4[0, 0], pair4[0, 1], 1 }, { pair5[0, 0], pair5[0, 1], 1 }, { pair6[0, 0], pair6[0, 1], 1 } };

double[] xVals1 = [pair1[1, 0], pair2[1, 0], pair3[1, 0]];
double[] yVals1 = [pair1[1, 1], pair2[1, 1], pair3[1, 1]];
double[] xVals2 = [pair4[1, 0], pair5[1, 0], pair6[1, 0]];
double[] yVals2 = [pair4[1, 1], pair5[1, 1], pair6[1, 1]];

var xDst1 = new List<double>();
var yDst1 = new List<double>();
var xDst2 = new List<double>();
var yDst2 = new List<double>();

Cv2.Solve(
    InputArray.Create(points1), InputArray.Create(xVals1),
    OutputArray.Create(xDst1),
    DecompTypes.LU);

Cv2.Solve(
    InputArray.Create(points1), InputArray.Create(yVals1),
    OutputArray.Create(yDst1),
    DecompTypes.LU);

Cv2.Solve(
    InputArray.Create(points2), InputArray.Create(xVals2),
    OutputArray.Create(xDst2),
    DecompTypes.LU);

Cv2.Solve(
    InputArray.Create(points2), InputArray.Create(yVals2),
    OutputArray.Create(yDst2),
    DecompTypes.LU);

// transform function 結果
Console.WriteLine($"Img1:");
Console.WriteLine("\ta: {0}, b: {1}, c: {2}, d: {3}, e: {4}, f: {5}", xDst1[0], xDst1[1], xDst1[2], yDst1[0], yDst1[1], yDst1[2]);

Console.WriteLine($"Img2:");
Console.WriteLine("\ta: {0}, b: {1}, c: {2}, d: {3}, e: {4}, f: {5}", xDst2[0], xDst2[1], xDst2[2], yDst2[0], yDst2[1], yDst2[2]);

using var img1 = new Mat(inputFiles[0], ImreadModes.Color);
using var img2 = new Mat(inputFiles[1], ImreadModes.Color);
using var img3 = new Mat(inputFiles[2], ImreadModes.Color);

var img2Mat = new Mat<Vec3b>(img2);
var img2Indexer = img2Mat.GetIndexer();

var img3Mat = new Mat<Vec3b>(img3);
var img3Indexer = img3Mat.GetIndexer();

var intermdiate = new Mat<Vec3b>(img2.Height + yPadding * 2, (int)Math.Round(img2.Width + img3.Width * 1.5));
var ma = xDst2[0];
var mb = xDst2[1];
var mc = xDst2[2];
var md = yDst2[0];
var me = yDst2[1];
var mf = yDst2[2];

// img3->img2 的 transform
for (int newY = -yPadding; newY < img2.Height + 2 * yPadding; newY++)
{
    for (int newX = 0; newX < img2.Width + img3.Width + xPadding; newX++)
    {
        var x = ma * newX + mb * newY + mc;
        var y = md * newX + me * newY + mf;

        if (x < 0 || x >= img3.Width || y < 0 || y >= img3.Height)
        {
            // Console.WriteLine($"Out of bounds: ({x}, {y})");
            continue;
        }

        Vec3b color = Interpolate(ref img3Indexer, x, y);
        intermdiate.Set(newY + yPadding, newX, color);
    }
}

for (int y = 0; y < img2.Height; y++)
{
    for (int x = 0; x < img2.Width; x++)
    {
        Vec3b color = img2Indexer[y, x];
        intermdiate.Set(y + yPadding, x, color);
    }
}

int maxX, maxY, minY;

FindBoundry(intermdiate, out maxX, out maxY, out minY);
intermdiate = intermdiate.SubMat(0, maxY, 0, maxX);

// Cv2.ImShow("intermediate", intermdiate);
// Cv2.WaitKey();
// Cv2.DestroyAllWindows();

var intermdiateIndexer = intermdiate.GetIndexer();

var outputH = Math.Max(img1.Height, intermdiate.Height) + yPadding * 2;
var outputW = img1.Width + intermdiate.Width + xPadding;
var finalOutput = new Mat<Vec3b>(outputH, outputW);

var a = xDst1[0];
var b = xDst1[1];
var c = xDst1[2];
var d = yDst1[0];
var e = yDst1[1];
var f = yDst1[2];

// intermdiate->img1 的 transform
for (int newY = -yPadding; newY < outputH; newY++)
{
    for (int newX = 0; newX < outputW; newX++)
    {
        // [h, w]
        var x = a * newX + b * newY + c;
        var y = d * newX + e * newY + f;
        if (y < 0 || y >= intermdiate.Height || x < 0 || x >= intermdiate.Width)
        {
            // Console.WriteLine($"Out of bounds: ({x}, {y})");
            continue;
        }

        Vec3b color = Interpolate(ref intermdiateIndexer, x, y);
        finalOutput.Set(newY, newX, color);
    }
}

var img1Indexer = new Mat<Vec3b>(img1).GetIndexer();
for (int y = 0; y < img1.Height; y++)
{
    for (int x = 0; x < img1.Width; x++)
    {
        Vec3b color = img1Indexer[y, x];
        finalOutput.Set(y + yPadding, x, color);
    }
}

FindBoundry(finalOutput, out maxX, out maxY, out minY);

finalOutput = finalOutput.SubMat(minY, maxY, 0, maxX);

// Cv2.ImShow("output", finalOutput);
// Cv2.WaitKey();
// Cv2.DestroyAllWindows();

// output file
finalOutput.SaveImage("output.jpg");

Console.WriteLine($"Output saved to output.jpg");


// bilinear interpolation
static Vec3b Interpolate(ref MatIndexer<Vec3b> matIndexer, double x, double y)
{
    int x1 = (int)Math.Floor(x);
    int x2 = x1 + 1;
    int y1 = (int)Math.Floor(y);
    int y2 = y1 + 1;

    double dx = x - x1;
    double dy = y - y1;

    Vec3b c1 = matIndexer[y1, x1];
    Vec3b c2 = matIndexer[y1, x2];
    Vec3b c3 = matIndexer[y2, x1];
    Vec3b c4 = matIndexer[y2, x2];

    Vec3b c = new();

    // r, g, b 3 channels
    for (int i = 0; i < 3; i++)
    {
        c[i] = (byte)(c1[i] * (1 - dx) * (1 - dy) + c2[i] * dx * (1 - dy) + c3[i] * (1 - dx) * dy + c4[i] * dx * dy);
    }

    return c;

}

/// <summary>
/// 取得圖片有效範圍
/// </summary>
static void FindBoundry(Mat<Vec3b> finalOutput, out int maxX, out int maxY, out int minY)
{
    maxX = 0;
    maxY = 0;
    minY = int.MaxValue;
    var outputIndexer = finalOutput.GetIndexer();

    // 裁切有效範圍
    for (int y = 0; y < finalOutput.Height; y++)
    {
        for (int x = 0; x < finalOutput.Width; x++)
        {
            var (i1, i2, i3) = outputIndexer[y, x];
            if (i1 != 0 || i2 != 0 || i3 != 0)
            {
                if (x > maxX)
                {
                    maxX = x;
                }
                if (y > maxY)
                {
                    maxY = y;
                }
                if (y < minY)
                {
                    minY = y;
                }
            }
        }
    }
}