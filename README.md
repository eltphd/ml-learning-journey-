# ML Learning Journey: From Statistical Learning to Modern AI

Documentation of my self-directed learning in modern machine learning, building on my PhD foundation in statistical modeling.

## 🎯 Learning Objective

**Bridge classical statistical learning (LCA/LTA, mixture models) with modern ML approaches (neural networks, transformers, RL) to enhance my research on interpretable AI for social systems.**

## 📚 Learning Path

### Phase 1: Foundations (Completed)
- ✅ **Andrew Ng's Machine Learning Specialization** (Coursera)
  - Supervised learning, neural networks, regularization
  - Gradient descent optimization
  - Bias-variance tradeoff
- ✅ **Andrew Ng's Deep Learning Specialization** (Coursera)
  - Neural network architectures
  - Convolutional and recurrent networks
  - Hyperparameter tuning, batch normalization

### Phase 2: NLP & Transformers (In Progress)
- 🔄 **Hugging Face NLP Course**
- 🔄 **Attention mechanisms & transformers**
- 🔄 **Fine-tuning pretrained models**

### Phase 3: Reinforcement Learning (Planned)
- 📋 **RL fundamentals** (Sutton & Barto)
- 📋 **Deep RL** (DQN, PPO, RLHF)
- 📋 **Multi-armed bandits** (for adaptive surveys)

## 🛠️ Projects & Implementations

### 1. Neural Network from Scratch
**Goal:** Understand backpropagation mechanics beyond sklearn's black box

- Implemented multi-layer perceptron in pure Python/NumPy
- Forward pass, backward pass, gradient descent
- Tested on classic datasets (MNIST, Iris)
- **Key insight:** Deepened understanding of optimization landscapes

[See: `01_neural_network_from_scratch/`]

### 2. Comparing Clustering Methods
**Goal:** Connect my LCA expertise to modern unsupervised learning

- Compared LCA (poLCA) vs. K-means vs. Gaussian Mixture Models
- Applied all three to same education dataset
- Evaluated: BIC, silhouette scores, interpretability
- **Key insight:** GMM is essentially probabilistic K-means, LCA adds categorical constraints

[See: `02_clustering_comparison/`]

### 3. Fine-Tuning LLM on Qualitative Data
**Goal:** Explore if transformers can capture participant voice patterns

- Fine-tuned DistilBERT on dissertation interview transcripts (n=47 adolescents)
- Task: Classify speaker turns by discrimination context
- Evaluated: Accuracy, F1, confusion matrix
- **Key insight:** Small models can capture domain-specific patterns, but require careful prompt engineering

[See: `03_llm_finetuning/`]

### 4. Gradient Descent Visualizations
**Goal:** Build intuition for optimization in high dimensions

- Implemented various optimizers: SGD, Momentum, Adam
- Visualized convergence on simple functions
- Experimented with learning rate schedules
- **Key insight:** Adaptive learning rates critical for non-convex surfaces

[See: `04_optimization_visualizations/`]

### 5. Interpretable ML for Education Data
**Goal:** Apply SHAP/LIME to understand model predictions

- Trained random forest on student dropout prediction
- Applied SHAP to explain individual predictions
- Compared with traditional logistic regression
- **Key insight:** Interpretability-accuracy tradeoff real, but explainable AI tools help

[See: `05_interpretable_ml/`]

## 📊 Key Concepts Mastered

### Optimization
- Gradient descent variants (SGD, mini-batch, Adam)
- Learning rate schedules
- Regularization (L1, L2, dropout)
- Batch normalization

### Neural Networks
- Activation functions (ReLU, sigmoid, tanh, softmax)
- Backpropagation algorithm
- Weight initialization strategies
- Overfitting prevention

### Model Evaluation
- Train/validation/test splits
- Cross-validation
- Precision, recall, F1, AUC-ROC
- Confusion matrices

### Unsupervised Learning
- K-means, hierarchical clustering
- Gaussian Mixture Models
- Dimensionality reduction (PCA, t-SNE, UMAP)
- Latent variable models

## 🔗 Connecting Statistical Learning to Modern ML

| Statistical Learning (My PhD) | Modern ML Equivalent | Connection |
|-------------------------------|----------------------|------------|
| Latent Class Analysis | Gaussian Mixture Models | Both are mixture models; GMM for continuous, LCA for categorical |
| Maximum Likelihood Estimation | Gradient Descent Optimization | Different optimization approaches for same goal |
| EM Algorithm | Variational Methods | Iterative optimization with latent variables |
| Information Criteria (BIC) | Validation Loss | Model selection balancing fit vs. complexity |
| Factor Analysis | Autoencoders | Both learn latent representations |
| Latent Transition Analysis | Recurrent Neural Networks | Both model temporal sequences |

## 🧪 Experiments & Insights

### Experiment 1: Can neural networks replicate LCA?
**Setup:** Train feedforward network to predict LCA class assignments  
**Result:** 87% accuracy, but less interpretable than LCA probabilities  
**Learning:** Neural networks great for prediction, LCA better for understanding subgroups

### Experiment 2: Transformers for survey response patterns
**Setup:** Use BERT embeddings for open-ended survey responses  
**Result:** Captured semantic themes missed by traditional coding  
**Learning:** NLP tools can augment qualitative analysis, not replace human interpretation

### Experiment 3: Multi-task learning for mental health outcomes
**Setup:** Jointly predict depression, anxiety, suicidality  
**Result:** Shared representations improved rare outcome detection  
**Learning:** Multi-task learning mirrors how clinicians assess co-occurring conditions

## 📖 Resources & References

### Online Courses
- [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction) - Andrew Ng
- [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) - Andrew Ng
- [Fast.ai Practical Deep Learning](https://course.fast.ai/)
- [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course)

### Books
- *Hands-On Machine Learning* - Aurélien Géron
- *Deep Learning* - Goodfellow, Bengio, Courville
- *Pattern Recognition and Machine Learning* - Christopher Bishop

### Papers I'm Reading
- "Attention Is All You Need" (Vaswani et al., 2017)
- "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2019)
- "Interpretable Machine Learning" - Molnar (online book)

## 🎓 What's Next

### Near-term (3-6 months)
- [ ] Implement transformer architecture from scratch
- [ ] Explore reinforcement learning for adaptive survey design
- [ ] Contribute to open-source ML libraries (scikit-learn, statsmodels)
- [ ] Write technical blog posts comparing statistical vs. ML approaches

### Medium-term (6-12 months)
- [ ] Apply for OpenAI Residency 2026
- [ ] Prototype Measurement Ally platform with LLM integration
- [ ] Submit paper: "Latent Transition Analysis Meets Deep Learning"
- [ ] Build portfolio of AI ethics consulting projects

## 💡 Philosophy

I'm not trying to become a pure ML engineer. My strength is the intersection:
- **Statistical rigor** from quantitative methods training
- **Domain expertise** in education equity and mental health
- **Interpretability focus** from working with real communities
- **Rapid learning** from building while studying

This combination—stats + ML + social impact—is my unique contribution to AI research.

## 📫 Contact

**Erica L. Tartt, PhD**  
📧 ericatartt@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/ericatartt)  
🐙 [GitHub](https://github.com/eltphd)

---

*"The best way to learn is to build. The best way to build is to learn from mistakes."*
