Model Selection

- once parameters w, b are fit to the training set, the training error J_error(w,b) is likely lower than the actual generalization error → average error on new examples not in the training set
- J_test(w,b) is better estimate of how well the model will generalize to new data compared to J_train(w,b)
- a procedure you could try is try models from degree 1 to 10, calculate all the J_test(w,b) and see which one gives the lowest value
- how well does the model perform → report test set error
- the problem: J_test(w,b) is likely to be an optimistic estimate of generalization error (ie J_test(w,b) < generalization error)
- this is because an extra parameter d (degree of polynomial) was chosen using the test set
- just like w, b are overly optimistic estimates of generalization error on training data

Training/cross validation/test set 

- split data into three different subsets
- 60% → training set / m_train
- 20% → cross validation / m_cv
- 20% → test set / m_test
- check the validity of different models
- also know as the validation set / development set / dev set
- can compute the training error, dev error, and test error

Model selection

- on degree 1 - 10 models, evaluation based on dev set
- then, to choose the model, look at model with the lowest dev error
- to estimate the generalization error, use the test set J_test(w,b)

Choosing a NN architecture

- to help decide how many layers should the NN have and how many units per layer → train many types of  models, eval on dev set, compute as fraction of misclassified examples
- pick model with lowest dev error
- estimate the generalization error with test set
- don’t look at the test set at all while making decisions regarding parameters with the model

Diagnosing bias and variance

- models almost never work on the first time
- looking at the bias and variance of a learning alg gives very good guidance on what to try next

- instead of looking at plots, a way to diagnose if the model has high bias or high variance would be to look at the performance of the alg on the training set and cross validation set
- high Jtrain → high bias
- high Jcv is much higher than Jtrain → high variance
- Jtrain is high on the left, Jcv is also high
- Jtrain is low on the right, Jcv is high
- Jtrain is low in the  middle, Jcv is low in the middle

Understanding bias and variance

- training error goes down with increased degree of polynomials
- cross validation error has a been curve for increased degree of polynomials

Diagnosing bias and variance

- High bias (underfit) → Jtrain will be high, Jtrain ~ Jcv
- High variance (overfit) → Jcv >> Jtarin, Jtrain may be low
- High bias and high variance → Jtrain will be high and Jcv >> Jtrain

Linear regression with regularization

- lamda → regularization parameter that controls how much you trade-off keeping the parameters w small vs fitting the training data well
- large lambda → high bias (underfit), Jtrain is large
- small lambda → high variance (overfit)

Choosing the regularization parameter lambda

- same process similar to choosing degree of polynomial with cross validation
- double lambda every time giving different parameters and cross validation error

Speech recognition example

- often good to compare training error to human level of recognition
- initially, a 10.8% error might lead to a high bias guess, but since the human performance is close to that percent and the cross validation error is almost 5% higher, then the alg has more of a variance problem

Establish a baseline level of performance

- what is the level of error you can reasonably hope to get to?
    - human level performance
    - competing alg performance
    - guess based on experience
- measure difference between baseline performance and training error, if high → high bias
- measure difference between training error and cross validation error, if high → high variance

Learning curves

- as the training set size increases, the cross validation error decreases
- as the training set size increases (and the degree of polynomial of the model stays the same), the training error increases → harder to fit all the training examples perfectly

High bias (underfitting)

- the average training error flattens with increased training set size
- cv error gets better and flattens out
- the model is too simple to fit to the examples better

- will never dip down to the human level performance
- more data will not help with high bias

High variance (overfitting)

- Jtrain can be better than human level performance

- if the learning alg suffers from high variance, getting more training data is likely to help
- to test, only use a little bit of the data and add a chunk at a time and track the gap between Jcv and Jtrain

Debugging a learning alg

- after implementing regularized linear regression on housing prices, the model makes unacceptably large errors in predictions, what to try next?
    - get more training examples → fixes higher variance
    - try smaller sets of features → fixes high variance
    - try getting additional features → fixes high bias
    - try adding polynomial features → fixes high bias
    - try decreasing lambda → fixes high bias
    - try increasing lambda → fixes high variance

The bias variance tradeoff

- fitting different order polynomials for linear models, for example, has a tradeoff between simple models that have high bias and a complex model that has high variance
- NN offer a way out of the dilemma

NN and bias variance

- large NN are low bias machines
- if the NN is large enough, you can almost always fit the training set well
    - first measure Jtrain (relative to baseline performance)
    - bad → bigger network (more hidden layers and/or units
    - good → does it do well on the cross validation set
        - bad → alg has high variance / more data (go to beginning with training set)
        - good → done

NN and regularization

- a large NN with, with well-chosen regularization, will usually do as well or better than a smaller one
- almost never hurt to go to a larger NN, so long as you regularize appropriately, but will slow down the training
- add extra term of kernel_regularizer=L2(0.01)

Iterative loop of ML development

- choose architecture (model, data, hyperparameters, etc.)
- train model
- diagnostics (bias, variance and error analysis)
- back to the start with new choice of architecture

Building a spam classifier

- Supervised learning
    - x → features of email
    - y → spam or not spam
- Features → list the top 10,000 words to compute x1, x2, …, x10000
- try to reduce the error
    - collect more data
    - develop sophisticated features based on email routing
    - define sophisticated features from email body
    - design algs to detect misspellings

Error analysis

- bias and variance most important, then error analysis
- manually looking through the misclassified examples and try to gain insights
- find examples and group them by common traits
- categories are not mutually exclusive
- get more data specific to error categories

Adding data

- tempting to always get ‘more data’ → slow and expensive
- better to add more data of the types where error analysis has indicated it might help
- go to unlabeled data and find more examples of certain types of data

Data augmentation

- modifying existing training example to create a new training example
- rotation, enlarging, shrinking, contrast, mirror
- distortion or transformation
- can also introduce warping of images

- for speech → noisy background or car or bad cellphone connection
- distortion introduced should be representative of the type of noise/distortions in the test set
- usually not helpful to add purely random/meaningless noise to the data

Data synthesis

- Artificial data synthesis for photo OCR → optical character recognition
- different fonts with contrasts

Engineering the data used by the system

- conventional model-centric approach: AI = code + data → working on code more
- data-centric approach: AI = code + data → working on data more

Transfer leaning: using data from a different task

- train algorithm on huge known data set
- to apply transfer learning, make a copy of the NN where you keep the parameters for all layers except the last, which gets swapped to the desired 10 for example, in handwritten classification
- so train the last layer parameters from scratch
- option 1: only train output layers parameters
- option 2: train all parameters → but every layer but the last are initialized with known parameters
- known as supervised pretraining and fine tuning
- downloading pretrained models

Why does transfer learning work?

- each layer learns features of the image to detect
- edges, corners, curves/basic shapes

- one restriction to supervised pretraining is that the image type x must be the same for the pretraining and fine tuning steps

Summary for transfer learning

- download NN parameters pretrained on a large dataset with same input type (images, audio, text) → one million images
- further train (fine tune) the network on your own data → 50 - 1000 images

Full cycle of a machine learning project

- scope project → define project
- collect data → define and collect data
- train model → training, error analysis and iterative improvement, may go back to collect data
- deploy in production → deploy, monitor and maintain system, go back to train with deploy data

Deployment

- mobile app makes api call to inference server with ML model on it and gets back an inference from the model on the server

- software engineering may be needed for
    - ensuring reliable and efficient predictions
    - scaling
    - logging
    - system monitoring
    - model updates
- MLOps → machine learning operations, do all these things

Fairness, bias, and ethics

- bias
    - hiring tool that discriminates against women
    - facial recognition system matching dark skinned individuals to criminal mugshots
    - biased bank loan approvals
    - toxic effect of reinforcing negative stereotypes
- adverse use cases
    - deepfakes
    - spreading toxic/incendiary speech through optimizing for engagement
    - generating fake content for commercial or political purposes
    - using ML to build harmful products, commit fraud
    - spam vs ani-spam: fraud vs anti-fraud

Guidelines

- get a diverse team to brainstorm things that might go wrong, with emphasis on possible harm to vulnerable groups
- carry out literature search on standards/guidelines for your industry
- audit systems against possible harm prior to deployment
- develop mitigation plan, and after deployment, monitor for possible harm → rollback to earlier version

Error metric for skewed datasets

- train classifier f_w,b(x) → find that you got 1% error on test set
- if this is a rare disease and y = 1 very rarely, that not so good
- if only 0.5% of the patients have the disease → print(”y=0”) does better at 0.5% error
- with skewed data sets, you can’t tell if 1% error is a good result or a bad result
- different error metric

Precision/recall

- y = 1 in presence of rare class we want to detect
- construct confusion matrix

- precision → of all patients where we predicted y=1, what fraction actually have the rare disease?)

- recall → of all patients that actually have the rare disease, what fraction did we correctly detect as having it?)

Trading off precision and recall

- suppose we want to predict y=1 only if very confident
- may choose a higher threshold → 70%, precision increases, recall decreases
- suppose we want to avoid missing too many cases of rare disease (when in doubt predict y=1)
- may choose a lower threshold → 30%, precision decreases, recall increases
- manually choose threshold

F1 score

- how to compare precision/recall numbers?
- combine into a single score
- average → doesn’t work that well
- F1 score → gives more emphasis to which value is lower, known as the harmonic mean