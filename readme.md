
# Deep Learning Notes

> **AI is the new electricity ⚡**  
Just like electricity transformed industries, AI is now transforming healthcare, transportation, manufacturing, communication, and more.

---

## Progression of AI

- **Rule-based algorithms** (if-else type logic)  
- **Machine Learning (ML)** algorithms  
- **Deep Learning (DL)** algorithms  
- **Transformers** (modern DL architectures for NLP & CV)  
- **Generative AI** (text/image/video generation)  
- **Explainable AI (XAI)** – making AI decisions understandable  

---

## What is Machine Learning (ML)?

**Definition (Tom Mitchell):**  
A computer program is said to learn from *experience (E)* with respect to *task (T)* and *performance (P)*, if its performance on **T** improves with **E** as measured by **P**.

👉 Example:  
- **Task (T):** Predict spam emails  
- **Experience (E):** Past labeled emails (spam or not)  
- **Performance (P):** Accuracy on test emails  

If accuracy improves as the system sees more data → it is learning.

---

## Learning Methods

- **Supervised Learning** – Learn from labeled data.  
  *Example:* Predicting house prices from past data.

- **Unsupervised Learning** – Learn from unlabeled data.  
  *Example:* Customer segmentation in marketing.

- **Reinforcement Learning** – Learn by trial and error.  
  *Example:* A robot learning to walk.

---

## Types of AI

- **Narrow AI (Weak AI):** Specialized in one task (e.g., Google Translate, Siri).  
- **AGI (Artificial General Intelligence):** Human-like intelligence (still research stage).  
- **Super AI:** Beyond human intelligence (theoretical).  
- **Generative AI (Gen AI):** Creates new content (ChatGPT, Midjourney).  
- **Explainable AI (XAI):** Focused on transparency of AI decisions.  

---

## Evolution of Deep Learning

Deep Learning evolved due to:

- 📊 Data availability (Big Data, images, text, videos)  
- ⚡ Computational power (GPUs, TPUs)  
- 🔬 Improved algorithms (CNN, RNN, Transformers)  

---

## Applications of Deep Learning

- **Medicine** → disease detection, X-rays, MRI  
- **Education** → intelligent tutoring systems  
- **Robotics** → autonomous movement, decision-making  
- **Agriculture** → crop monitoring, pest detection  
- **Climate Change** → predicting weather patterns  
- **Gaming** → realistic NPCs, game bots  
- **Finance** → credit scoring, fraud detection  
- **Text Generation** → chatbots, summarization  
- **Image Generation** → AI art, design  

---

## AI and Its Usage

- **Natural Language Processing (NLP):** Chatbots, translation  
- **Computer Vision (CV):** Image recognition, object detection  
- **Speech & Signals:** Speech-to-text, voice assistants  

---

## How Machines Learn?

Think of a simple **linear equation**:

y = m . x + b

- \(x\) = input  
- \(y\) = output  
- \(m\) = slope (learnable parameter)  
- \(b\) = intercept (bias)  

 ML tries to find the best values of \(m\) and \(b\) so predictions are close to actual outputs.

**Process:**
1. Take training data (inputs + correct outputs).  
2. Fit a best-fit line/curve.  
3. Use that model for predictions on unseen data.  

---

## Training & Prediction Process

**🔹 Training Phase:**  
- Input training data.  
- Model adjusts weights (parameters) to minimize error.  
- Algorithm used: **Gradient Descent**.  

**🔹 Prediction Phase:**  
- New unseen data given.  
- Model uses learned weights to predict output.  

*Example:* Predicting future sales after training on past sales data.  

---

## AI vs ML vs DL (The Hierarchy)

- **Artificial Intelligence (AI):** The broadest field. Goal = mimic human intelligence (problem-solving, learning, planning).  
- **Machine Learning (ML):** A subset of AI. Systems learn patterns from data using algorithms.  
- **Deep Learning (DL):** A subset of ML. Uses **Artificial Neural Networks (ANNs)** with many layers to learn complex patterns.  

---

## ML Concepts

### Encoding Categorical Data

ML models work with **numbers**, not text.  
Categorical data must be encoded.

- **Label Encoding** – Assigns each category a number.  

```text
Fruits = [Apple, Banana, Orange]
Label Encode → [0, 1, 2]
````

* **One-Hot Encoding** – Creates binary columns for each category.

```text
Fruits = [Apple, Banana, Orange]
One-Hot Encode →
Apple  Banana  Orange
  1      0      0
  0      1      0
  0      0      1
```

✅ Rule of thumb:

* Use **Label Encoding** if categories are ordered (e.g., Small < Medium < Large).
* Use **One-Hot Encoding** if categories are unordered (e.g., colors, fruits).

---

### Scaling Data

**Why scale data?**
Different features may have different ranges.
Example:

* Age = 20–80
* Income = 10,000–100,000

Without scaling → features with larger values dominate.
Scaling ensures fair contribution of features.

⚖️ **Two Common Methods:**

1. **Normalization (Min-Max Scaling):**

$$
x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
$$

* Scales data to a fixed range \[0, 1].
* Use when you know **min** and **max**.

*Example (Age 20–80):*

$$
x = 20 \Rightarrow \frac{20-20}{80-20} = 0
$$

$$
x = 80 \Rightarrow \frac{80-20}{80-20} = 1
$$

---

2. **Standardization (Z-score Scaling):**

$$
x' = \frac{x - \mu}{\sigma}
$$

where:

* $\mu$ = mean of feature

* $\sigma$ = standard deviation

* Transforms data to:

  * Mean = 0
  * Std Dev = 1

*Example (Height \~ mean 170, std 10):*

$$
x = 180 \Rightarrow \frac{180 - 170}{10} = 1
$$

---

💡 **When to Scale?**
Scale data before using algorithms sensitive to feature scale:

* ✅ K-Nearest Neighbors (KNN)
* ✅ Support Vector Machines (SVM)
* ✅ Logistic/Linear Regression (with gradient descent)
* ✅ Neural Networks


## Regression

Regression predicts **continuous values**.

### Simple Linear Regression


y = w . x + b


*Example:* Predict house price from size.

---

### Multiple Regression

Multiple Regression:  

$$
y = w_1x_1 + w_2x_2 + \dots + w_nx_n + b
$$


*Example:* Predict house price from size, bedrooms, and location.

 **Output:** Any real number (not a category).

---

## Classification

Classification predicts **categories (discrete values).**

- **Binary Classification:** Two classes (e.g., spam / not spam).  
- **Multi-class Classification:** More than two classes (e.g., types of fruit).  

 **Algorithms:** Logistic Regression, Decision Trees, SVM, Neural Networks.  

---

## Performance Metrics

Used to evaluate **classification models**.

### Confusion Matrix Terms

- **TP (True Positive):** Predicted Positive & actually Positive  
- **TN (True Negative):** Predicted Negative & actually Negative  
- **FP (False Positive):** Predicted Positive but actually Negative  
- **FN (False Negative):** Predicted Negative but actually Positive  

---


## 📐 Formulas

---

### 1. Accuracy

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

✔️ Good when classes are **balanced**.

---

### 2. Precision (Positive Predictive Value)

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

_(How many predicted positives are actually correct.)_

---

### 3. Recall (Sensitivity)

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

_(How many actual positives we correctly identified.)_

---

### 4. F1 Score (Balances Precision & Recall)

$$
F1 = \frac{2 \cdot (\text{Precision} \cdot \text{Recall})}{\text{Precision} + \text{Recall}}
$$



---

## Visualization

Helps to **understand and explain ML results**.

- **Scatter Plots:** Show relation between two variables.  
  *Example:* Study hours vs Exam score.  

- **Confusion Matrix (Table):**

|               | Predicted Pos | Predicted Neg |
|---------------|---------------|---------------|
| **Actual Pos** | TP            | FN            |
| **Actual Neg** | FP            | TN            |

Helps visualize model errors.

 Other useful plots: **ROC curve**, **Precision-Recall curve** (advanced but important).  



### ANN Introduction

- Versatile and Powerful: They are the core of DL and can solve highly complex tasks that are difficult for traditional ML (image recognition, speech recognition, recommendation systems, game-playing AI).

History: First proposed in 1943 (McCulloch & Pitts) as a model of biological neurons using logic.

#### Why Now? Three reasons for their recent explosion:

- Big Data: Huge datasets are available to train them.
- More Computing Power: GPUs can train large networks quickly.
- Algorithmic Improvements: We've found ways to train them effectively (e.g., overcoming the fear of "local optima").

### Biological Neuron

 The Brain's Building Block

- This is the biological system we are loosely inspired by.

Parts:
 - Dendrites receive signals. 
 - The cell body processes them. 
 - The axon transmits a signal if the total input is strong enough.
 - Synapses are the connections to other neurons' dendrites.

Function: 

A neuron "fires" (sends an electrical signal) if it receives enough signals from other connected neurons.

 Complex computation emerges from networks of billions of these simple connected units.

 #### Structure:

- Dendrites → receive signals.
- Cell body (soma) → nucleus.
- Axon → long extension.
- Synapses → connections to next neurons.

#### Working:

- Neuron receives electrical impulses.
- If signals strong → neuron fires.
- Billions of neurons → complex computations.


## Logical Computations with Neurons

### 🧠 Core Idea: The Threshold Logic Unit (TLU)

McCulloch and Pitts proposed a simplified mathematical model of a biological neuron, called a **Threshold Logic Unit (TLU)** or an **Artificial Neuron**.

A TLU works on a simple rule:

1. It calculates a **weighted sum** of its inputs.  
2. It compares this sum to a predefined **threshold value** ($\theta$).  
3. If the sum **≥ θ**, the neuron "fires" (outputs 1).  
4. If the sum **< θ**, it does not fire (outputs 0).  

---

### ✏️ Mathematical Representation

$$
\text{Output} =
\begin{cases}
1 & \text{if } (w_1x_1 + w_2x_2 + \dots + w_nx_n) \geq \theta \\
0 & \text{otherwise}
\end{cases}
$$

**Where:**

- $x_1, x_2, \dots, x_n$: input values (usually binary: 0 or 1)  
- $w_1, w_2, \dots, w_n$: corresponding weights (connection strengths)  
- $\theta$: threshold value (determines if the neuron fires)

---

## Logic Gates with a TLU

### 1. AND Operation ($C = A \land B$)

The AND gate outputs **1 only if both inputs are 1**.

| Input A ($x_1$) | Input B ($x_2$) | Output |
|----------------|----------------|--------|
| 0              | 0              | 0      |
| 0              | 1              | 0      |
| 1              | 0              | 0      |
| 1              | 1              | 1      |

**TLU settings:**  
- $w_1 = 1$, $w_2 = 1$, $\theta = 2$

$$
y = (1 \cdot x_1) + (1 \cdot x_2)
$$

- (0,0): 0 ≥ 2 → 0  
- (0,1): 1 ≥ 2 → 0  
- (1,0): 1 ≥ 2 → 0  
- (1,1): 2 ≥ 2 → 1 ✅  

✔️ **Conclusion**: A single TLU with these parameters performs **AND**.

---

### 2. OR Operation ($C = A \lor B$)

The OR gate outputs **1 if at least one input is 1**.

| Input A ($x_1$) | Input B ($x_2$) | Output |
|----------------|----------------|--------|
| 0              | 0              | 0      |
| 0              | 1              | 1      |
| 1              | 0              | 1      |
| 1              | 1              | 1      |

**TLU settings:**  
- $w_1 = 1$, $w_2 = 1$, $\theta = 1$

$$
y = (1 \cdot x_1) + (1 \cdot x_2)
$$

- (0,0): 0 ≥ 1 → 0  
- (0,1): 1 ≥ 1 → 1  
- (1,0): 1 ≥ 1 → 1  
- (1,1): 2 ≥ 1 → 1 ✅  

✔️ **Conclusion**: Changing only the threshold lets the TLU perform **OR**.

---

### 3. NOT Operation ($C = \lnot A$)

The NOT gate has **one input** and outputs the **opposite value**.

| Input A ($x_1$) | Output |
|----------------|--------|
| 0              | 1      |
| 1              | 0      |

**TLU settings:**  
- $w_1 = -1$, $\theta = 0$

$$
y = -1 \cdot x_1
$$

- Input 0: (−1 × 0) = 0 ≥ 0 → 1  
- Input 1: (−1 × 1) = −1 ≥ 0 → 0 ✅  

✔️ **Conclusion**: With a **negative weight**, the TLU performs **NOT**.

---

Nice! You're building a solid explainer on **Perceptrons** — and this version is already rich in detail. Let's clean it up and format it for proper **Markdown with LaTeX support** so that:

* Math renders cleanly using `$...$` or `$$...$$`
* Lists and sections are structured clearly
* It's easy to read on GitHub or in your notes

---



## The Perceptron (Architecture)

### 1. 🧠 The Big Idea: From Theory to Practice

**Who & When:** Frank Rosenblatt, 1957.  
**What it is:** The Perceptron wasn't just a mathematical model (like the McCulloch–Pitts neuron); it was the first algorithmically defined, **trainable** artificial neural network. It was even implemented in hardware ("the Mark I Perceptron machine").

**Significance:**  
It's the direct ancestor of all modern neural networks. It took the simple logic-gate neuron and made it capable of **learning from data**.

---

### 2. 🧱 The Building Block: Linear Threshold Unit (LTU)

The Perceptron is built from one or more **LTUs** — a more advanced version of the McCulloch–Pitts neuron.

#### What the LTU Does:

1. **Receives Inputs:**  
   Takes in multiple **real-valued inputs**: $x_1, x_2, ..., x_n$  
   _(e.g., pixel intensity, exam score, sensor reading)_

2. **Computes Weighted Sum:**  
   Each input has a weight $w_1, w_2, ..., w_n$.  
   - Positive weight ⟶ positive influence  
   - Negative weight ⟶ negative influence  
   - Near-zero weight ⟶ weak influence  

   The LTU calculates:

   $$
   z = (w_1 \cdot x_1) + (w_2 \cdot x_2) + \dots + (w_n \cdot x_n) + b
   $$

   Or compactly, using vector notation:

   $$
   z = \mathbf{w}^T \cdot \mathbf{x}
   $$

   - $\mathbf{w} = [w_1, w_2, ..., w_n]$ (weight vector)  
   - $\mathbf{x} = [x_1, x_2, ..., x_n]$ (input vector)  
   - Bias $b$ can be included as $w_0$ with fixed input $x_0 = 1$

3. **Applies Activation Function:**  
   The weighted sum $z$ is passed through a **step function** to produce the final binary output.

   $$
   \hat{y} = h_w(x) = \text{step}(z)
   $$

---

### ⚙️ The Step Functions (Decision Makers)

#### 1. Heaviside Step Function

- If $z < 0$, output 0  
- If $z \geq 0$, output 1

$$
\text{heaviside}(z) = 
\begin{cases}
0 & \text{if } z < 0 \\
1 & \text{if } z \geq 0
\end{cases}
$$

**Use Case:** Binary classification with labels 0 and 1 (e.g., Spam vs. Not Spam)

---

#### 2. Sign Function (sgn)

- If $z < 0$, output -1  
- If $z = 0$, output 0  
- If $z > 0$, output +1

$$
\text{sgn}(z) =
\begin{cases}
-1 & \text{if } z < 0 \\
\ \ 0 & \text{if } z = 0 \\
+1 & \text{if } z > 0
\end{cases}
$$

**Use Case:** Binary classification with labels -1 and +1. Often more convenient mathematically.

---

### 🖼️ Visualizing the LTU


    x1 ----> w1 ----
    x2 ----> w2 ---- (+)
    x3 ----> w3 ----
                     |--> z = w1x1 + w2x2 + w3x3 + b --> step(z) --> ŷ
```

bias (x0 = 1) --> w0 --

```

- **Inputs:** $x_1, x_2, x_3$  
- **Weights:** $w_1, w_2, w_3$  
- **Bias:** $w_0$  
- **Summation node (Σ):** Calculates $z$  
- **Activation function:** Applies step decision

---

### 🧩 How It Works as a Classifier

The Perceptron (a single LTU) acts as a **linear binary classifier**.

- It finds a **decision boundary** in the input space:
  - Line (2D)
  - Plane (3D)
  - Hyperplane (nD)

- The weighted sum $z$ defines this boundary.

- The step function checks **which side of the boundary** a point $\mathbf{x}$ is on:

  - If $z \geq 0$ ⟶ Output 1 (or +1)  
  - If $z < 0$ ⟶ Output 0 (or -1)

✔️ If the dataset is **linearly separable**, the Perceptron can learn weights $w$ and bias $b$ to perfectly separate the classes.


