# Pneumonia-Detection-Using-Deep-Learning-A-Guide-to-Image-Classification
Deep learning project

# Pneumonia Dataset
The dataset used for this project is a curated collection of chest X-ray images organized into three main folders: train, test, and validation, each containing images categorized
as either ”Pneumonia” or ”Normal.” It comprises a total of5,863 high-resolution JPEG images, distributed across two categories: Pneumonia (positive cases) and Normal (negative cases).
This collection of images originates from pediatric patients, aged one to five years, treated at the Guangzhou Women and Children’s Medical Center in Guangzhou, China. The images
are anterior-posterior (AP) views of the chest, a standard radiographic technique that facilitates detailed examination of lung conditions.
To ensure data quality, all X-ray images underwent an initial screening for readability and resolution. Low-quality or ambiguous scans were excluded to avoid inaccuracies in
model training. Expert grading further validated the dataset, with two certified radiologists independently diagnosing each image. In cases where disagreements arose, a third specialist
provided a final assessment to confirm the diagnosis. This rigorous validation process ensures a high level of confidence in the labeled categories, forming a robust dataset for training
and testing deep learning models in pneumonia detection.

# process

The project aimed to develop a deep learning model for early pneumonia detection from chest X-ray images using Convolutional Neural Networks (CNNs) and transfer learning. The dataset consisted of labeled images categorized as "Pneumonia" and "Normal." The challenge was to distinguish pneumonia from other respiratory conditions, as they share similar features. To address this, data preprocessing was done, including image resizing, normalization, and augmentation (rotations, flips, zooming) to improve model generalization and reduce overfitting.
VGG19 and Xception models were explored. Initially, VGG19 showed lower accuracy and higher validation loss, possibly due to its inability to handle the complexity of chest X-ray images. Xception, with its advanced depthwise separable convolutions, outperformed VGG19, achieving higher accuracy and lower loss. Fine-tuning the Xception model, including unfreezing the last layers and adjusting the learning rate, improved performance. Techniques like early stopping, Adam and Adamax optimizers, and learning rate schedulers further enhanced model stability.
The successful Xception model was then deployed as a user-friendly web application using Streamlit, allowing healthcare professionals to upload chest X-rays and receive real-time pneumonia classifications. This practical tool demonstrates the potential of deep learning in medical imaging, aiding rapid, automated diagnostics, especially in resource-constrained environments.
