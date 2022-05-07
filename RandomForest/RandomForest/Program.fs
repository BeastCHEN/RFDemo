// Learn more about F# at http://docs.microsoft.com/dotnet/fsharp

open System
open System.IO
open Accord
open Accord.MachineLearning
open Accord.MachineLearning.DecisionTrees
open Accord.Math.Optimization.Losses
open Accord.Statistics
//let data = @"E:\RFDemo\iris.csv"
let data = @"E:\RFDemo\Filterd_Feature.csv"

let readData fileName =
    File.ReadAllLines fileName
    |> fun line -> line.[1..]
    |> Array.map (fun line -> line.Split(','))
    |> Array.map (fun line -> (line.[20] |> Convert.ToInt32), (line.[0..19] |> Array.map Convert.ToDouble))
    |> Array.unzip

let label, feature = readData data

let teacher =
    RandomForestLearning(NumberOfTrees = 1000)

let forest = teacher.Learn(feature, label)
let predictedLaber = forest.Decide(feature)

let error =
    ZeroOneLoss(label).Loss(forest.Decide(feature))

printf "%A" error
