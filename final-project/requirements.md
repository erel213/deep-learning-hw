# Final Project 2025 – Semester B  
**Deep Learning Course - Reichman University**

---

## Option 2: Guided Project – Pneumonia Classification from Chest X-Rays

### Task
Classify chest X-ray images as **normal** or **pneumonia** using the publicly available dataset:  
[Chest X-Ray Pneumonia Dataset (Kaggle)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

- The dataset is already split into `train`, `validation`, and `test` folders.
- **Leave the test set untouched** and report your final model performance only on this set.
- You may adjust the split between training and validation as you wish.

---

### What You Need to Do

1. **CNN Model**
   - Implement a Convolutional Neural Network to solve the classification task.

2. **Vision Transformer (ViT) Model**
   - Implement a ViT model based on the architecture and concepts from the paper:  
     [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929)  
   - You can experiment with parameters such as patch size, embedding dimension, etc.

3. **Model Comparison**
   - Compare and analyze the performance of the two models.

---

### Analysis Requirements

Your analysis should include:

- **Quantitative Metrics:**
  - Accuracy
  - Precision
  - Recall
  - F1-score

- **Training Insights:**
  - Training curves and convergence behavior
  - Generalization performance

- **Model Complexity:**
  - Training time
  - Number of parameters

- **Observations:**
  - Overfitting / Underfitting
  - Strengths and weaknesses of each model for this task

- **Additional Techniques (optional but encouraged):**
  - Data augmentation
  - Transfer learning
  - Other training optimizations
  - Evaluate their impact on results

- **Relevant Literature:**
  - For example:  
    *When Vision Transformers Outperform ResNets Without Pre-Training or Strong Data Augmentations*  
    [Paper link](https://arxiv.org/pdf/2106.01548)

---

### Submission Guidelines

Submit a **ZIP folder** containing:

1. **Report**  
   - Use the following Overleaf template:  
     [Project Report Template](https://www.overleaf.com/read/ysyygwggwbmr#392432)  
   - Maximum 8 pages (shorter reports are welcome).

2. **Code**  
   - Provide a link to your code repository (GitHub or Google Colab).
   - Code must be:
     - Clearly documented
     - Easy to follow
   - Frameworks allowed: **Keras, PyTorch, TensorFlow**

---

### Important Dates

- **August 17, 2025** – Final submission deadline

---

**Good luck!**
