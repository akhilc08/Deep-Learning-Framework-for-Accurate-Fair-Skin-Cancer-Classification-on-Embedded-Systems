# Deep-Learning-Framework-for-Accurate-Fair-Skin-Cancer-Classification-on-Embedded-Systems
This project introduces a deep learning framework for dermatology that balances accuracy, fairness, and latency on low-power devices. The approach tackles data scarcity, demographic bias, and hardware constraints by combining synthetic data generation with knowledge distillation, resulting in a compact yet accurate skin cancer classifier.

Highlights

- Data Augmentation: Balanced dataset across skin tones using Gaussian White Noise and GAN-generated synthetic images.

- Knowledge Distillation: Transferred insights from a large Swin Transformer teacher to a lightweight MobileNet student, reducing model size by 18× while retaining accuracy.

- Fairness-Oriented: Improved statistical parity fairness score by 12 points, reducing disparities across skin-tone subgroups.

- Low Latency: Achieved <1 second inference per image on a Raspberry Pi 4, enabling offline deployment in resource-constrained clinical settings.

Results

- Accuracy improved from 63% → 76% after distillation.

- Fairness score increased by 12 points with augmentation.

- Compact 6 MB MobileNet model vs. 114 MB Swin Transformer, with only minor drops in performance.

- Best configuration (GAN + Gaussian Noise, 4000 images/class) achieved an overall performance score of 0.88 (balanced across accuracy, fairness, and latency).
