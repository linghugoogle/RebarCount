//
//  ViewController.swift
//  RebarCount
//
//  Created by linghugoogle on 2025/9/30.
//

import UIKit

class ViewController: UIViewController {
    
    var imageView: UIImageView!
    var detectButton: UIButton!
    
    private var mnnHelper: MNNHelper?

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .white
        
        setupUI()
        setupMNNHelper()
        loadTestImage()
    }
    
    private func setupUI() {
        // 创建图像视图
        imageView = UIImageView()
        imageView.contentMode = .scaleAspectFit
        imageView.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(imageView)
        
        // 创建检测按钮
        detectButton = UIButton(type: .system)
        detectButton.setTitle("检测目标", for: .normal)
        detectButton.titleLabel?.font = UIFont.systemFont(ofSize: 18)
        detectButton.backgroundColor = .systemBlue
        detectButton.setTitleColor(.white, for: .normal)
        detectButton.layer.cornerRadius = 8
        detectButton.translatesAutoresizingMaskIntoConstraints = false
        detectButton.addTarget(self, action: #selector(detectButtonTapped), for: .touchUpInside)
        view.addSubview(detectButton)
        
        // 设置约束
        NSLayoutConstraint.activate([
            imageView.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 20),
            imageView.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
            imageView.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),
            imageView.heightAnchor.constraint(equalTo: view.heightAnchor, multiplier: 0.7),
            
            detectButton.topAnchor.constraint(equalTo: imageView.bottomAnchor, constant: 20),
            detectButton.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            detectButton.widthAnchor.constraint(equalToConstant: 120),
            detectButton.heightAnchor.constraint(equalToConstant: 44)
        ])
    }
    
    private func setupMNNHelper() {
        guard let modelPath = Bundle.main.path(forResource: "yolo11n_rebar", ofType: "mnn") else {
            print("无法找到模型文件")
            return
        }
        
        mnnHelper = MNNHelper(modelPath: modelPath)
    }
    
    private func loadTestImage() {
        if let image = UIImage(named: "0C7CB7B9.jpg") {
            imageView.image = image
        }
    }
    
    @objc private func detectButtonTapped() {
        guard let image = imageView.image,
              let helper = mnnHelper else {
            print("图像或MNN助手未准备好")
            return
        }
        
        detectButton.isEnabled = false
        detectButton.setTitle("检测中...", for: .normal)
        
        DispatchQueue.global(qos: .userInitiated).async {
            // 执行目标检测
            let results = helper.detectObjects(in: image)
            
            // 在检测结果上绘制边界框
            let resultImage = helper.drawDetectionResults(results, on: image)
            
            DispatchQueue.main.async {
                self.imageView.image = resultImage
                self.detectButton.isEnabled = true
                self.detectButton.setTitle("检测目标", for: .normal)
                
                print("检测到 \(results.count) 个目标")
            }
        }
    }
}

