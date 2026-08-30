Carrying Out Error Analysis

- manually looking at mistakes the model makes
- Look at dev examples to evaluate ideas
    - features specific to the wrong examples
    - Error analysis
        - get ~100 mislabeled dev set examples
        - count up how many groups
        - which features focus would get the most return
- Evaluate multiple ideas in parallel
    - create a table
    - percentage of each error categories

    - both false negatives and false positives

Cleaning Up Incorrectly Labeled Data

- Incorrectly labeled examples
    - mislabeled example → from model
    - incorrectly labeled → from source
    - DL algs are quite robust to random error in the training set
    - if random, okay to leave as they are
    - so long the data set is big enough
    - not robust to systematic errors → if all of one type of examples are labeled wrong
- Error analysis
    - in dev or test set → add extra column
    - if the error percent of incorrectly labeled data is significant → then change
    - goal of dev set is to help select between two classifiers A and B
- Correcting incorrect dev/test set examples
    - apply same process to the dev and test sets to make sure they continue to come from the same distribution
    - consider examining examples the alg got right as well as ones it got wrong
    - train and dev/test data may now come from slightly different distributions

Build your First System Quickly, then Iterate

- Speech recognition example
    - noisy background
        - cafe noise
        - cat noise
    - accented speech
    - far from mic
    - young children’s speech
    - stuttering
- set up dev/test set and metric
- build initial system quickly
- use bias/variance analysis and error analysis to prioritize next steps

Training and Testing on Different Distributions

- shove as much data into the training set even if the distribution is different from the dev and test set
- dilemma → not much data from true test data set, much a lot from other sources
- option 1
    - put both datasets together
    - shuffle evenly
    - disadvantage is most of dev set will not be from the correct output distribution
    - means the optimization target it training data distribution
    - option not recommended
- option 2
    - dev and test set are all user side, output distribution images
    - only a little added to the training data
    - now the target is set properly for a correct distribution
    - this option is better
- Speech recognition example
    - Training
        - purchased data
        - smart speaker control
        - voice keyboard
    - Dev/test
        - speech activated rearview mirror

Bias and Variance with Mismatched Data Distributions

- bias and variance analysis changes when the training and dev set come from different distributions
- Cat classifier example
    - if training error is 1%
    - if dev error is 10%
    - seems to be a variance problem and the alg isn’t generalizing well
    - but since the training data and dev data come from different distributions, can no longer draw the variance conclusion
    - dev set could contain more difficult images
    - two things changed at the same time, unknown examples in dev set and distribution different
    - training-dev set: same distribution as training, but not used for training
    - evaluate error analysis then on training-dev set, dev set, and test set
    - difference between training error and training-dev error→ variance
    - difference is between training-dev set and dev error → data mismatch problem
- Bias/variance on mismatched training and dev/test sets
    - human level
    - training set error
    - training-dev set error
    - dev error
    - test error

    - doesn’t always go up
    - dev/test set could be easier
- More general formulation

Addressing Data Mismatch

- not great systematic ways
- third potential problem
- carry out manual error analysis to try tot understand difference between training and dev/tests sets
- make training data more similar; or collect more data similar to dev/test set
- e.g. simulate noisy data
- Artificial data synthesis
    - reverb
    - car noise
    - data augmentation
    - if audio noise is stretched to all audio input, model could overfit to particular noise chosen

Transfer Learning

- knowledge from one model to another
- delete the last layer, new weights for new task
- retrain all the layers or just the new last layer
- also known as pretraining if all parameters are retrained
- fine-tuning is just the last layer
- low level features learns the structure of recognizing images
- or add even more layers
- makes sense if not a lot of data on new task, but a lot from another task
- not the other way around
- makes sense when task a and b have the same input x
- low level features from a could be helpful for learning b

Multi-task Learning

- transfer learning is a sequential process, learn from a and transfer to b
- multi-task → simultaneously try NN on several things
- Simplifies autonomous driving example
    - identifying multiple objects in an image
    - yhat is multi-output
    - loss changes to sum of loss for every yhat output
    - unlike softmax regression: one image can have multiple labels

    - could have made four NN
    - but one network to do four things is better than four NN
    - even if not all objects are labeled → just sum only over values of j with 0/1 label
- When multi-task learning makes sense
    - training on a set of tasks that could benefit from having shared lower-level features
    - usually: amount of data for each task is quite similar
    - training on many other tasks’ data could give a big boost to the current task, shared knowledge
    - can train a big enough NN to do well on all the tasks
    - only hurts if NN isn’t big enough

What is End-to-End Deep Learning?

- replace multiple stages of processing and replace it with usually just a single NN
- What is end-to-end learning?
    - MFCC → extracting hand designed features
    - audio → features → phonemes → words → transcript
    - with NN, audio → transcript
    - challenges, need a lot of data
- Face recognition
    - learn function mapping from person image to identity → not the best approach
    - problem is the person can approach from many directions and orientations
    - multi-step approach
        - detect person face
        - zoom in to face / crop
        - then fed into NN
        - to train on faces, train between two faces, same or not
        - then compare to all faces
        - less data for end to end image → identity
        - so better accuracy with two steps
    - More examples
        - machine translation
        - image → bones → age, there is enough data
        - image to age needs too much data

Whether to use End-to-end Deep Learning

- Pros and cons of end-to-end deep learning
    - pros
        - let the data speak
        - captures the statistics of the data rather than to be forces to reflect human preconceptions
        - phonemes are a human artifact, model can discover a better way to represent sounds
        - less hand-designing of components needed
    - cons
        - may need large amount of data
        - excludes potentially useful hand-designed components
        - data and hand-design → two components of model knowledge
- Applying end-to-end deep learning
    - key question: do you have sufficient data to learn a function of the complexity needed to map x to y
    - machine learning or deep learning for individual components
    - carefully check X → Y depending what tasks you can get data for

Rusian Salakhutdinov Interview

- Boltzmann machines
- contrasting divergence
- Restricted Boltzmann machines
- non-linear extension of PCA
- auto-encoders
- deep Boltzmann machines
- generative, supervised, semi-supervised, unsupervised
- generative adversarial networks
- variational auto-encoders
- deep energy models