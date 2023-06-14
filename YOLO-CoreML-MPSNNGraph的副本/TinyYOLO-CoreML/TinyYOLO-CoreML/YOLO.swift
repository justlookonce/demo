import Foundation
import UIKit
import CoreML

class YOLO {
    public static let inputWidth = 352
    public static let inputHeight = 608
    public static let maxBoundingBoxes = 10
    
    // Tweak these values to get more or fewer predictions.
    let confidenceThreshold: Float = 0.5
    let iouThreshold: Float = 0.4
    
    struct Prediction {
        let classIndex: Int
        let score: Float
        let rect: CGRect
    }
    //let model = best_3()
    //let model = TinyYOLO()
    //let mlConfig = MLModelConfiguration()
    var model = PredictPerson()
    
//    public init() {
//        mlConfig.computeUnits = .all
//        do {
//            model = try PredictPerson(configuration: mlConfig)
//        } catch {
//
//        }
//    }
    
    public func predict(image: CVPixelBuffer) -> [Prediction]? {
        if let output = try? model.prediction(images: image) {
            return computeBoundingBoxes_(features1: output._340,features2: output._338,features3: output.output)
                   }
        else {
            return nil
        }
    }
    //这个函数不重要，看下面那个
    public func computeBoundingBoxes(features: MLMultiArray) -> [Prediction] {
        var predictions = [Prediction]()
        assert(features.count==18*38*22)
        //for k in 0..<3{
        let gridHeight = 38
        let gridWidth = 22
        let blockSize_x: Float = 16
        let blockSize_y: Float = 16
        /*
         if (k==1){
         
         
         
         gridHeight = 38
         gridWidth = 22
         blockSize_x = 16
         blockSize_y = 16
         features = features2!
         if (features==nil){break}          }
         if (k==2){
         
         
         gridHeight = 38
         gridWidth = 22
         blockSize_x = 8
         blockSize_y = 8
         features = features3!
         if (features==nil){break}
         }
         */
        let boxesPerCell = 3
        let numClasses = 1
        
        // The 416x416 image is divided into a 13x13 grid. Each of these grid cells
        // will predict 5 bounding boxes (boxesPerCell). A bounding box consists of
        // five data items: x, y, width, height, and a confidence score. Each grid
        // cell also predicts which class each bounding box belongs to.
        //
        // The "features" array therefore contains (numClasses + 5)*boxesPerCell
        // values for each grid cell, i.e. 125 channels. The total features array
        // contains 125x13x13 elements.
        
        // NOTE: It turns out that accessing the elements in the multi-array as
        // `features[[channel, cy, cx] as [NSNumber]].floatValue` is kinda slow.
        // It's much faster to use direct memory access to the features.
        
        /*
         @inline(__always) func offset(_ channel: Int, _ x: Int, _ y: Int) -> Int {
         return channel*channelStride + y*yStride + x*xStride
         }*/
        
        for cy in 0..<gridHeight {
            for cx in 0..<gridWidth {
                for b in 0..<boxesPerCell {
                    
                    // For the first bounding box (b=0) we have to read channels 0-24,
                    // for b=1 we have to read channels 25-49, and so on.
                    let channel = b*(numClasses + 5)
                    
                    // The slow way:
                    
                    let tx = features[[0,channel    , cy, cx] as [NSNumber]].floatValue
                    let ty = features[[0,channel + 1, cy, cx] as [NSNumber]].floatValue
                    let tw = features[[0,channel + 2, cy, cx] as [NSNumber]].floatValue
                    let th = features[[0,channel + 3, cy, cx] as [NSNumber]].floatValue
                    let tc = features[[0,channel + 4, cy, cx] as [NSNumber]].floatValue
                    
                    
                    // The fast way:
                    /*
                     let tx = Float(featurePointer[offset(channel    , cx, cy)])
                     let ty = Float(featurePointer[offset(channel + 1, cx, cy)])
                     let tw = Float(featurePointer[offset(channel + 2, cx, cy)])
                     let th = Float(featurePointer[offset(channel + 3, cx, cy)])
                     let tc = Float(featurePointer[offset(channel + 4, cx, cy)])
                     print(featurePointer)*/
                    
                    // The predicted tx and ty coordinates are relative to the location
                    // of the grid cell; we use the logistic sigmoid to constrain these
                    // coordinates to the range 0 - 1. Then we add the cell coordinates
                    // (0-12) and multiply by the number of pixels per grid cell (32).
                    // Now x and y represent center of the bounding box in the original
                    // 416x416 image space.
                    let xt = (Float(cx) + tx * 2 - 0.5)
                    let x = xt * blockSize_x
                    let yt = (Float(cy) + ty * 2 - 0.5)
                    let y = yt * blockSize_y
                    // The size of the bounding box, tw and th, is predicted relative to
                    // the size of an "anchor" box. Here we also transform the width and
                    // height into the original 416x416 image space.
                    let w = 4 * pow(tw,2) * anchors[b] * blockSize_x
                    let h = 4 * pow(th,2) * anchors[b] * blockSize_y
                    
                    // The confidence value for the bounding box is given by tc. We use
                    // the logistic sigmoid to turn this into a percentage.
                    let confidence = tc
                    
                    // Gather the predicted classes for this anchor box and softmax them,
                    // so we can interpret these numbers as percentages.
                    //var classes = [Float](repeating: 0, count: numClasses)
                    //for c in 0..<numClasses {
                    // The slow way:
                    //classes[c] = features[[channel + 5 + c, cy, cx] as [NSNumber]].floatValue
                    
                    // The fast way:
                    //classes[c] = Float(featurePointer[offset(channel + 5 + c, cx, cy)])
                    //}
                    //classes = softmax(classes)
                    
                    let c = 1
                    let classes = features[[0,channel + 4 + c, cy, cx] as [NSNumber]].floatValue
                    //var classes = Float(featurePointer[offset(channel + 4 + c, cx, cy)])
                    
                    
                    // Find the index of the class with the largest score.
                    //let (detectedClass, bestClassScore) = classes.argmax()
                    //let (detectedClass, bestClassScore) = classes.
                    
                    // Combine the confidence score for the bounding box, which tells us
                    // how likely it is that there is an object in this box (but not what
                    // kind of object it is), with the largest class prediction, which
                    // tells us what kind of object it detected (but not where).
                    let confidenceInClass = classes * confidence
                    
                    // Since we compute 13x13x5 = 845 bounding boxes, we only want to
                    // keep the ones whose combined score is over a certain threshold.
                    if confidenceInClass > confidenceThreshold {
                        let rect = CGRect(x: CGFloat(x - w/2), y: CGFloat(y - h/2),
                                          width: CGFloat(w), height: CGFloat(h))
                        
                        let prediction = Prediction(classIndex: 0,
                                                    score: confidenceInClass,
                                                    rect: rect)
                        predictions.append(prediction)
                        
                    }
                }
            }
        }
        
        // We already filtered out any bounding boxes that have very low scores,
        // but there still may be boxes that overlap too much with others. We'll
        // use "non-maximum suppression" to prune those duplicate bounding boxes.
        return nonMaxSuppression(boxes: predictions, limit: YOLO.maxBoundingBoxes, threshold: iouThreshold)
    }
    
    
    
    
    
    
    public func computeBoundingBoxes_(features1: MLMultiArray,features2:MLMultiArray,features3:MLMultiArray) -> [Prediction] {
        var predictions = [Prediction]()
        for k in 0..<3{
            var gridHeight = 19
            var gridWidth = 11
            var blockSize_x: Float = 32
            var blockSize_y: Float = 32
            var features = features1
            var anchors_: [Float] = [3.625, 2.8125, 4.875, 6.1875, 11.65625, 10.1875]
            if (k==1){
                
                
                
                gridHeight = 38
                gridWidth = 22
                blockSize_x = 16
                blockSize_y = 16
                features = features2
                anchors_ = [1.875, 3.8125, 3.875, 2.8125, 3.6875, 7.4375]
            }
            if (k==2){
                
                
                gridHeight = 76
                gridWidth = 44
                blockSize_x = 8
                blockSize_y = 8
                features = features3
                anchors_ = [1.25, 1.625, 2.0, 3.75, 4.125, 2.875]
                
            }
            
            let boxesPerCell = 3
            let numClasses = 1
            
            // The 416x416 image is divided into a 13x13 grid. Each of these grid cells
            // will predict 5 bounding boxes (boxesPerCell). A bounding box consists of
            // five data items: x, y, width, height, and a confidence score. Each grid
            // cell also predicts which class each bounding box belongs to.
            //
            // The "features" array therefore contains (numClasses + 5)*boxesPerCell
            // values for each grid cell, i.e. 125 channels. The total features array
            // contains 125x13x13 elements.
            
            // NOTE: It turns out that accessing the elements in the multi-array as
            // `features[[channel, cy, cx] as [NSNumber]].floatValue` is kinda slow.
            // It's much faster to use direct memory access to the features.
            
            /*
             @inline(__always) func offset(_ channel: Int, _ x: Int, _ y: Int) -> Int {
             return channel*channelStride + y*yStride + x*xStride
             }*/
            
            for cy in 0..<gridHeight {
                for cx in 0..<gridWidth {
                    for b in 0..<boxesPerCell {
                        
                        // For the first bounding box (b=0) we have to read channels 0-24,
                        // for b=1 we have to read channels 25-49, and so on.
                        let channel = b*(numClasses + 5)
                        
                        // The slow way:
                        
                        let tx = features[[0,channel    , cy, cx] as [NSNumber]].floatValue
                        let ty = features[[0,channel + 1, cy, cx] as [NSNumber]].floatValue
                        let tw = features[[0,channel + 2, cy, cx] as [NSNumber]].floatValue
                        let th = features[[0,channel + 3, cy, cx] as [NSNumber]].floatValue
                        let tc = features[[0,channel + 4, cy, cx] as [NSNumber]].floatValue
                        
                        
                        // The fast way:
                        /*
                         let tx = Float(featurePointer[offset(channel    , cx, cy)])
                         let ty = Float(featurePointer[offset(channel + 1, cx, cy)])
                         let tw = Float(featurePointer[offset(channel + 2, cx, cy)])
                         let th = Float(featurePointer[offset(channel + 3, cx, cy)])
                         let tc = Float(featurePointer[offset(channel + 4, cx, cy)])
                         print(featurePointer)*/
                        
                        // The predicted tx and ty coordinates are relative to the location
                        // of the grid cell; we use the logistic sigmoid to constrain these
                        // coordinates to the range 0 - 1. Then we add the cell coordinates
                        // (0-12) and multiply by the number of pixels per grid cell (32).
                        // Now x and y represent center of the bounding box in the original
                        // 416x416 image space.
                        let xt = (Float(cx) + tx * 2 - 0.5)
                        let x = xt * blockSize_x
                        let yt = (Float(cy) + ty * 2 - 0.5)
                        let y = yt * blockSize_y
                        // The size of the bounding box, tw and th, is predicted relative to
                        // the size of an "anchor" box. Here we also transform the width and
                        // height into the original 416x416 image space.
                        let w = 4 * pow(tw,2) * anchors_[2*b] * blockSize_x
                        let h = 4 * pow(th,2) * anchors_[2*b+1] * blockSize_y

                        // The confidence value for the bounding box is given by tc. We use
                        // the logistic sigmoid to turn this into a percentage.
                        let confidence = tc
                        
                        // Gather the predicted classes for this anchor box and softmax them,
                        // so we can interpret these numbers as percentages.
                        //var classes = [Float](repeating: 0, count: numClasses)
                        //for c in 0..<numClasses {
                        // The slow way:
                        //classes[c] = features[[channel + 5 + c, cy, cx] as [NSNumber]].floatValue
                        
                        // The fast way:
                        //classes[c] = Float(featurePointer[offset(channel + 5 + c, cx, cy)])
                        //}
                        //classes = softmax(classes)
                        
                        var classes = [Float](repeating: 0, count: numClasses)
                        for c in 0..<numClasses {
                          classes[c] = features[[0,channel + 4 + c, cy, cx] as [NSNumber]].floatValue
                        }
                        classes = softmax(classes)

                        // Find the index of the class with the largest score.
                        let (detectedClass, bestClassScore) = classes.argmax()

                        
                        
//                        let c = 1
//                        let classes = features[[0,channel + 4 + c, cy, cx] as [NSNumber]].floatValue
                        //var classes = Float(featurePointer[offset(channel + 4 + c, cx, cy)])
                        
                        
                        // Find the index of the class with the largest score.
                        //let (detectedClass, bestClassScore) = classes.argmax()
                        //let (detectedClass, bestClassScore) = classes.
                        
                        // Combine the confidence score for the bounding box, which tells us
                        // how likely it is that there is an object in this box (but not what
                        // kind of object it is), with the largest class prediction, which
                        // tells us what kind of object it detected (but not where).
                        let confidenceInClass = bestClassScore * confidence
                        
                        // Since we compute 13x13x5 = 845 bounding boxes, we only want to
                        // keep the ones whose combined score is over a certain threshold.
                        if confidenceInClass > confidenceThreshold {
                            let rect = CGRect(x: CGFloat(x - w/2), y: CGFloat(y - h/2),
                                              width: CGFloat(w), height: CGFloat(h))
                            
                            let prediction = Prediction(classIndex: detectedClass,
                                                        score: confidenceInClass,
                                                        rect: rect)
                            predictions.append(prediction)
                            
                        }
                    }
                }
            }
            
        }
                
                // We already filtered out any bounding boxes that have very low scores,
                // but there still may be boxes that overlap too much with others. We'll
                // use "non-maximum suppression" to prune those duplicate bounding boxes.
                return nonMaxSuppression(boxes: predictions, limit: YOLO.maxBoundingBoxes, threshold: iouThreshold)
        }
    }

