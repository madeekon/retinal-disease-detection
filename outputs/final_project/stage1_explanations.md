# Final Project - Stage 1: concise explanations for the manuscript

## MLP architecture
The multilayer perceptron receives the 1024-dimensional DenseNet121 embedding of one fundus image,
standardized with the mean and standard deviation of the training subset. Each hidden layer computes an
affine transformation followed by a nonlinear activation (ReLU or tanh) and, optionally, dropout. The output
layer is a single neuron that returns one raw score (logit) for the class "Abnormal". Four configurations
were compared: one hidden layer of 256 units without dropout (A), with dropout 0.5 (B), two hidden layers of
512 and 128 units with dropout 0.3 (C), and one hidden layer of 256 tanh units with dropout 0.3 (D).

## Forward pass
For layer l: z(l) = W(l) h(l-1) + b(l), h(l) = phi(z(l)), with h(0) = x. The output logit
z = w^T h(L) + b is converted to a probability p = sigmoid(z) = 1 / (1 + e^(-z)). An image is classified as
Abnormal when p >= 0.5.

## Loss function
Binary cross-entropy: L = -[y log p + (1 - y) log(1 - p)], averaged over the batch. PyTorch's
BCEWithLogitsLoss combines the sigmoid and the cross-entropy in one numerically stable function. For this
pair the gradient with respect to the output logit is simply p - y.

## Backpropagation
After the forward pass the error at the output, delta(L) = p - y, is propagated backwards with the chain
rule: delta(l) = (W(l+1)^T delta(l+1)) * phi'(z(l)). The gradient of each weight matrix is
dL/dW(l) = delta(l) h(l-1)^T and of each bias dL/db(l) = delta(l). Dropout multiplies the same random mask
in the forward and backward pass. PyTorch computes these gradients automatically (loss.backward()).

## Optimizers
* Full-batch gradient descent: one update per epoch using the gradient over all training samples.
  Exact gradient, but very few updates for the same data budget.
* Stochastic gradient descent (batch 1): one update per sample. Many cheap but noisy updates; the noise can
  help escape poor regions but makes the loss curve unstable and is slow in wall-clock time.
* Mini-batch gradient descent (batch 64): a compromise; the gradient is averaged over 64 samples, giving
  about 70 updates per epoch with moderate noise and efficient vectorized computation.
* SGD with momentum (0.9): keeps an exponentially weighted average of past gradients (velocity),
  v = 0.9 v + g, w = w - lr v, which damps oscillations and speeds up progress along consistent directions.
* Adam: adapts a separate step size for each weight from running estimates of the first and second moments
  of the gradient (with bias correction), so it needs little learning-rate tuning and converges quickly.
All methods received the same budget of 60 epochs (the same number of presented samples); the learning rate
of each method was chosen on the validation loss only.

## Key stage-1 results (test set, mean ± std over 3 seeds; full table: stage1_summary_test.csv)
* SVM-RBF (C = 1, gamma = 1e-4, deterministic): F1 0.716, ROC-AUC 0.766. Random Forest: F1 0.715 ± 0.008, ROC-AUC 0.755 ± 0.003.
* MLP configurations (Adam, lr 1e-4): F1 between 0.720 and 0.727, ROC-AUC 0.767-0.772. Depth, activation and dropout
  change the result by less than 0.01 F1, i.e. within about two standard deviations. Configuration D (tanh, dropout 0.3)
  had the highest mean validation F1 (0.725) and was selected; on the test set it reached F1 0.720 ± 0.006.
* Optimizers (same 60-epoch budget): all reach a similar best validation loss (0.536-0.545) and test F1 0.712-0.727.
  They differ in how they get there: full-batch GD needs about 32 epochs (32 updates) and overfits least; mini-batch
  methods reach their best after 7-10 epochs and then overfit; SGD with batch 1 reaches its best after about 2 epochs,
  drives the training loss almost to zero, is the least stable and is about 17 times slower in wall-clock time
  (175 s vs 10 s).
* Overall, nonlinear models on frozen DenseNet121 embeddings improve only slightly over Logistic Regression
  (Assignment 03, test F1 0.710): the embedding, not the classifier, limits performance.

## Notes for honesty
* The first MLP run used Adam lr 1e-3; the validation loss was lowest after 1-5 epochs (fast overfitting). The
  learning rate was lowered to 1e-4 based on validation loss only (the same value won the Adam search in script 07).
* For mini-batch GD, momentum and Adam the selected learning rate was the smallest value of the grid, so a slightly
  smaller rate might be marginally better.
* Training times were measured on a CPU in the analysis environment and depend on the computer.
