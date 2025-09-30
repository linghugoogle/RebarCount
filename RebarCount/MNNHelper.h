//
//  MNNHelper.h
//  RebarCount
//
//  Created by linghugoogle on 2025/9/30.
//

#import <UIKit/UIKit.h>
#import <CoreGraphics/CoreGraphics.h>

NS_ASSUME_NONNULL_BEGIN

// 检测结果结构体
typedef struct {
    CGRect bbox;
    float confidence;
    int classId;
} DetectionResult;

@interface MNNHelper : NSObject

// 初始化方法，传入模型路径
- (instancetype)initWithModelPath:(NSString *)modelPath;

// 检测图片中的目标
- (NSArray<NSValue *> *)detectObjectsInImage:(UIImage *)image;

// 在图片上绘制检测框
- (UIImage *)drawDetectionResults:(NSArray<NSValue *> *)results onImage:(UIImage *)image;

@end

NS_ASSUME_NONNULL_END
