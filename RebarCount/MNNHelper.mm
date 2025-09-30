//
//  MNNHelper.m
//  RebarCount
//
//  Created by linghugoogle on 2025/9/30.
//

#import "MNNHelper.h"
#import <MNN/MNNDefine.h>
#import <MNN/expr/Module.hpp>
#import <MNN/expr/Executor.hpp>
#import <MNN/expr/ExprCreator.hpp>
#import <MNN/cv/cv.hpp>
#import <MNN/ImageProcess.hpp>
#import <vector>
#import <memory>

using namespace MNN;
using namespace MNN::Express;
using namespace MNN::CV;

@interface MNNHelper()
@property (nonatomic, strong) NSString *modelPath;
@property (nonatomic, assign) std::shared_ptr<Module> net;
@property (nonatomic, assign) std::shared_ptr<Executor::RuntimeManager> rtmgr;
@end

@implementation MNNHelper

- (instancetype)initWithModelPath:(NSString *)modelPath {
    self = [super init];
    if (self) {
        self.modelPath = modelPath;
        [self setupMNNModel];
    }
    return self;
}

- (void)setupMNNModel {
    // 配置MNN运行时
    MNN::ScheduleConfig sConfig;
    sConfig.type = MNN_FORWARD_CPU;
    sConfig.numThread = 4;
    
    MNN::BackendConfig bConfig;
    bConfig.precision = MNN::BackendConfig::Precision_Normal;
    sConfig.backendConfig = &bConfig;
    
    // 创建运行时管理器
    self.rtmgr = std::shared_ptr<Executor::RuntimeManager>(
        Executor::RuntimeManager::createRuntimeManager(sConfig)
    );
    
    if (self.rtmgr == nullptr) {
        NSLog(@"Failed to create RuntimeManager");
        return;
    }
    
    // 加载模型
    const char* modelPathCStr = [self.modelPath UTF8String];
    self.net = std::shared_ptr<Module>(
        Module::load(std::vector<std::string>{}, std::vector<std::string>{}, modelPathCStr, self.rtmgr)
    );
    
    if (self.net == nullptr) {
        NSLog(@"Failed to load model from path: %@", self.modelPath);
    } else {
        NSLog(@"Model loaded successfully");
    }
}

- (NSArray<NSValue *> *)detectObjectsInImage:(UIImage *)image {
    if (self.net == nullptr) {
        NSLog(@"Model not loaded");
        return @[];
    }
    
    // 将UIImage转换为MNN可处理的格式
    VARP inputVar = [self preprocessImage:image];
    if (inputVar == nullptr) {
        NSLog(@"Failed to preprocess image");
        return @[];
    }
    
    // 执行推理
    auto outputVar = self.net->forward(inputVar);
    if (outputVar == nullptr) {
        NSLog(@"Failed to run inference");
        return @[];
    }
    
    // 后处理检测结果
    return [self postprocessOutput:outputVar imageSize:image.size];
}

- (VARP)preprocessImage:(UIImage *)image {
    // 获取图片的CGImage
    CGImageRef cgImage = image.CGImage;
    if (!cgImage) {
        return nullptr;
    }
    
    // 获取图片尺寸
    size_t width = CGImageGetWidth(cgImage);
    size_t height = CGImageGetHeight(cgImage);
    
    NSLog(@"Original image size: %zu x %zu", width, height);
    
    // 创建位图上下文
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    unsigned char *rawData = (unsigned char*)malloc(height * width * 4);
    NSUInteger bytesPerPixel = 4;
    NSUInteger bytesPerRow = bytesPerPixel * width;
    NSUInteger bitsPerComponent = 8;
    
    CGContextRef context = CGBitmapContextCreate(rawData, width, height,
                                                bitsPerComponent, bytesPerRow, colorSpace,
                                                kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
    CGColorSpaceRelease(colorSpace);
    
    CGContextDrawImage(context, CGRectMake(0, 0, width, height), cgImage);
    CGContextRelease(context);
    
    // 计算缩放和padding
    int ih = (int)height;
    int iw = (int)width;
    int len = ih > iw ? ih : iw;
    float scale = 640.0f / len;
    
    // 直接创建640x640的输入tensor (NCHW格式)
    std::vector<int> input_dims = {1, 3, 640, 640}; // [batch, channels, height, width]
    auto input_var = _Input(input_dims, NCHW, halide_type_of<float>());
    auto input_ptr = input_var->writeMap<float>();
    
    // 初始化为0 (padding区域)
    memset(input_ptr, 0, 1 * 3 * 640 * 640 * sizeof(float));
    
    // 计算缩放后的尺寸
    int new_h = (int)(ih * scale);
    int new_w = (int)(iw * scale);
    
    // 计算padding偏移
    int pad_h = (640 - new_h) / 2;
    int pad_w = (640 - new_w) / 2;
    
    NSLog(@"Scaled size: %d x %d, padding: (%d, %d)", new_w, new_h, pad_w, pad_h);
    
    // 手动进行双线性插值缩放和RGB转换
    for (int y = 0; y < new_h; y++) {
        for (int x = 0; x < new_w; x++) {
            // 计算原图坐标
            float src_x = x / scale;
            float src_y = y / scale;
            
            int x0 = (int)src_x;
            int y0 = (int)src_y;
            int x1 = std::min(x0 + 1, (int)width - 1);
            int y1 = std::min(y0 + 1, (int)height - 1);
            
            float dx = src_x - x0;
            float dy = src_y - y0;
            
            // 双线性插值
            for (int c = 0; c < 3; c++) {
                float val00 = rawData[(y0 * width + x0) * 4 + c] / 255.0f;
                float val01 = rawData[(y0 * width + x1) * 4 + c] / 255.0f;
                float val10 = rawData[(y1 * width + x0) * 4 + c] / 255.0f;
                float val11 = rawData[(y1 * width + x1) * 4 + c] / 255.0f;
                
                float val = val00 * (1 - dx) * (1 - dy) + 
                           val01 * dx * (1 - dy) + 
                           val10 * (1 - dx) * dy + 
                           val11 * dx * dy;
                
                // 存储到NCHW格式: [batch][channel][height][width]
                int dst_y = y + pad_h;
                int dst_x = x + pad_w;
                if (dst_y >= 0 && dst_y < 640 && dst_x >= 0 && dst_x < 640) {
                    input_ptr[c * 640 * 640 + dst_y * 640 + dst_x] = val;
                }
            }
        }
    }
    
    free(rawData);
    
    // 验证tensor形状
    auto info = input_var->getInfo();
    NSLog(@"Input tensor shape: [%d, %d, %d, %d]", 
          info->dim[0], info->dim[1], info->dim[2], info->dim[3]);
    
    return input_var;
}

- (NSArray<NSValue *> *)postprocessOutput:(VARP)output imageSize:(CGSize)imageSize {
    NSMutableArray *results = [NSMutableArray array];
    
    if (output == nullptr) {
        return results;
    }
    
    // 转换输出格式
    output = _Convert(output, NCHW);
    output = _Squeeze(output, {0}); // 移除batch维度
    
    // 获取输出信息
    auto info = output->getInfo();
    if (!info) {
        NSLog(@"Failed to get output info");
        return results;
    }
    
    auto dims = info->dim;
    if (dims.size() != 2) {
        NSLog(@"Unexpected output dimensions: %zu", dims.size());
        return results;
    }
    
    int numFeatures = dims[0]; // 84 (4 + 80 classes)
    int numAnchors = dims[1];  // 8400
    
    NSLog(@"Output shape: [%d, %d]", numFeatures, numAnchors);
    
    auto outputPtr = output->readMap<float>();
    if (!outputPtr) {
        NSLog(@"Failed to read output data");
        return results;
    }
    
    std::vector<DetectionResult> detections;
    
    // 计算缩放和padding参数
    float ih = imageSize.height;
    float iw = imageSize.width;
    float len = ih > iw ? ih : iw;
    float scale = 640.0f / len;  // 从640缩放回原图的比例
    
    // 计算padding偏移
    int pad_h = (640 - (int)(ih * scale)) / 2;
    int pad_w = (640 - (int)(iw * scale)) / 2;
    
    NSLog(@"Scale: %f, Padding: (%d, %d)", scale, pad_w, pad_h);
    
    // 遍历所有anchor点
    for (int i = 0; i < numAnchors; i++) {
        // 获取边界框坐标 (中心点格式，相对于640x640像素坐标)
        float cx = outputPtr[0 * numAnchors + i];
        float cy = outputPtr[1 * numAnchors + i];
        float w = outputPtr[2 * numAnchors + i];
        float h = outputPtr[3 * numAnchors + i];
        
        // 找到最大置信度的类别
        float maxProb = 0.0f;
        int maxClassId = 0;
        for (int c = 4; c < numFeatures; c++) {
            float prob = outputPtr[c * numAnchors + i];
            if (prob > maxProb) {
                maxProb = prob;
                maxClassId = c - 4;
            }
        }
        
        // 置信度阈值过滤
        if (maxProb > 0.25f) {
            NSLog(@"Detection %d: cx=%f, cy=%f, w=%f, h=%f, conf=%f, class=%d", 
                  i, cx, cy, w, h, maxProb, maxClassId);
            
            // 转换为左上角坐标格式 (相对于640x640)
            float x0 = cx - w * 0.5f;
            float y0 = cy - h * 0.5f;
            float x1 = cx + w * 0.5f;
            float y1 = cy + h * 0.5f;
            
            // 去除padding影响
            x0 -= pad_w;
            y0 -= pad_h;
            x1 -= pad_w;
            y1 -= pad_h;
            
            // 缩放回原图尺寸
            x0 /= scale;
            y0 /= scale;
            x1 /= scale;
            y1 /= scale;
            
            // 边界检查
            x0 = std::max(0.0f, std::min(iw, x0));
            y0 = std::max(0.0f, std::min(ih, y0));
            x1 = std::max(0.0f, std::min(iw, x1));
            y1 = std::max(0.0f, std::min(ih, y1));
            
            // 计算宽高
            float width = x1 - x0;
            float height = y1 - y0;
            
            // 确保宽高大于0
            if (width > 0 && height > 0) {
                DetectionResult detection;
                detection.bbox = CGRectMake(x0, y0, width, height);
                detection.confidence = maxProb;
                detection.classId = maxClassId;
                
                NSLog(@"Valid detection: x=%f, y=%f, w=%f, h=%f", x0, y0, width, height);
                detections.push_back(detection);
            }
        }
    }
    
    NSLog(@"Found %zu detections before NMS", detections.size());
    
    // 应用NMS
    std::vector<DetectionResult> nmsResults = [self performNMS:detections iouThreshold:0.45f];
    
    NSLog(@"Found %zu detections after NMS", nmsResults.size());
    
    // 转换为NSValue数组
    for (const auto& detection : nmsResults) {
        NSValue *value = [NSValue valueWithBytes:&detection objCType:@encode(DetectionResult)];
        [results addObject:value];
    }
    
    return results;
}

- (std::vector<DetectionResult>)performNMS:(std::vector<DetectionResult>)detections iouThreshold:(float)threshold {
    std::vector<DetectionResult> result;
    
    // 按置信度排序
    std::sort(detections.begin(), detections.end(), [](const DetectionResult& a, const DetectionResult& b) {
        return a.confidence > b.confidence;
    });
    
    std::vector<bool> suppressed(detections.size(), false);
    
    for (size_t i = 0; i < detections.size(); i++) {
        if (suppressed[i]) continue;
        
        result.push_back(detections[i]);
        
        for (size_t j = i + 1; j < detections.size(); j++) {
            if (suppressed[j]) continue;
            
            float iou = [self calculateIOU:detections[i].bbox rect2:detections[j].bbox];
            if (iou > threshold) {
                suppressed[j] = true;
            }
        }
    }
    
    return result;
}

- (float)calculateIOU:(CGRect)rect1 rect2:(CGRect)rect2 {
    CGRect intersection = CGRectIntersection(rect1, rect2);
    if (CGRectIsEmpty(intersection)) {
        return 0.0f;
    }
    
    float intersectionArea = intersection.size.width * intersection.size.height;
    float union_area = (rect1.size.width * rect1.size.height) + 
                      (rect2.size.width * rect2.size.height) - intersectionArea;
    
    return intersectionArea / union_area;
}

- (UIImage *)drawDetectionResults:(NSArray<NSValue *> *)results onImage:(UIImage *)image {
    if (results.count == 0) {
        return image;
    }
    
    // 将NSValue数组转换为DetectionResult数组以便排序
    NSMutableArray<NSValue *> *sortedResults = [NSMutableArray arrayWithCapacity:results.count];
    
    // 先提取所有DetectionResult
    NSMutableArray<NSDictionary *> *detectionData = [NSMutableArray arrayWithCapacity:results.count];
    for (int i = 0; i < results.count; i++) {
        NSValue *value = results[i];
        DetectionResult detection;
        [value getValue:&detection];
        
        // 存储原始NSValue和对应的y坐标、x坐标用于排序
        NSDictionary *item = @{
            @"value": value,
            @"y": @(detection.bbox.origin.y),
            @"x": @(detection.bbox.origin.x)
        };
        [detectionData addObject:item];
    }
    
    // 按照从上到下、从左到右的顺序排序
    [detectionData sortUsingComparator:^NSComparisonResult(NSDictionary *obj1, NSDictionary *obj2) {
        CGFloat y1 = [obj1[@"y"] floatValue];
        CGFloat y2 = [obj2[@"y"] floatValue];
        CGFloat x1 = [obj1[@"x"] floatValue];
        CGFloat x2 = [obj2[@"x"] floatValue];
        
        // 首先按y坐标排序（从上到下）
        if (fabs(y1 - y2) > 20.0) { // 允许20像素的容差，认为是同一行
            return y1 < y2 ? NSOrderedAscending : NSOrderedDescending;
        }
        
        // 如果y坐标相近，按x坐标排序（从左到右）
        return x1 < x2 ? NSOrderedAscending : NSOrderedDescending;
    }];
    
    // 提取排序后的NSValue数组
    for (NSDictionary *item in detectionData) {
        [sortedResults addObject:item[@"value"]];
    }
    
    UIGraphicsBeginImageContextWithOptions(image.size, NO, image.scale);
    [image drawAtPoint:CGPointZero];
    
    CGContextRef context = UIGraphicsGetCurrentContext();
    
    int detectionIndex = 1; // 从1开始编号
    
    for (NSValue *value in sortedResults) {
        DetectionResult detection;
        [value getValue:&detection];
        
        // 设置绘制样式
        CGContextSetStrokeColorWithColor(context, [UIColor redColor].CGColor);
        CGContextSetLineWidth(context, 2.0);
        
        // 绘制边界框
        CGContextStrokeRect(context, detection.bbox);
        
        // 计算中心点
        CGFloat centerX = detection.bbox.origin.x + detection.bbox.size.width / 2.0;
        CGFloat centerY = detection.bbox.origin.y + detection.bbox.size.height / 2.0;
        
        // 绘制数字标记
        NSString *numberLabel = [NSString stringWithFormat:@"%d", detectionIndex];
        NSDictionary *numberAttributes = @{
            NSForegroundColorAttributeName: [UIColor whiteColor],
            NSFontAttributeName: [UIFont boldSystemFontOfSize:24]
        };
        
        // 计算文字尺寸以居中显示
        CGSize textSize = [numberLabel sizeWithAttributes:numberAttributes];
        CGPoint textPoint = CGPointMake(centerX - textSize.width / 2.0, 
                                       centerY - textSize.height / 2.0);
        
        [numberLabel drawAtPoint:textPoint withAttributes:numberAttributes];
        
        detectionIndex++;
    }
    
    UIImage *resultImage = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    
    return resultImage;
}

@end
