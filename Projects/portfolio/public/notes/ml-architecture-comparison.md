# Notables

## Pytorch vs TensorFlow
- first thing I noted is the similarity between these two architectures
- the most glaring differences were the training loop and switching the toggling on and off trainable parameters throughout training
- creating the actual model is pretty much identical, when using Sequential, only slightly different syntax
- after some research, when creating custom models and training loops, that's where tensorflow's gradient tape really would shine, but I did not bump into the gradient tape in this project
- pytorch, I did note, was easier to use and understand after looking back at the code
- one huge difference, as of May 2026, was that tensorflow code does not run on NVIDIA GPU (CUDA) on Windows, which was suprising to me, although I've overheard thay accessability will return soon to Windows
- as a workaround I used WSL (Windows Subsystem for Linux), which was interesting to learn about
- another big difference was the stark contrast in accurcies I got on the same pretrained models with the same hyper
- the pytorch implementation (the side that I optimized for hypers) received close to 90% acc, while the same setup with tensorflow initially was consistently in the lower 80s or even high 70s
- my reasoning, after some research, came down to model weight initializations, optimizer implementations, and how tensors are handled on the gpu (although the same calculations are being done with either architecture so I'm still not certain on this point)

## Results
- an interesting point I noted was the difference in accuracies per rerun
- over the course of this project, I had to rerun training on all the models many times and saw different accuracies, sometimes quite significant (+/- one percent), across different runs with the same config
- I would like to chalk that up to weight initialization and the randomness of the batch orders in training, but I could never pinpoint the reason
- for example, on my final rerun, a scratch CNN run with 0.7 dropout performed the best as opposed to most other runs, of the same config, having 0.5 dropout rate performing the best
- one surprising result was how well (again, based on the static best config from the scratch cnn section) the EfficientNet performed

## Scikit-Learn
- for the sklearn section, I wanted to experiment with different popular sklearn classifiers
- the way I could implement this was to take the final feature vector from the pretrained mocels and use that as the input for the classifiers
- since the output of the pre-trained models had such rich final feature vectors, the classifiers had no problem training on those data
- surpisingly, the classifiers performed as well as fine tuning the models themselves
- I reasoning is because the pre-trained models are just that good at providing unique rich feature vectors at model's end
- I added some more visuals to the sklearn section just to fill up space, a good place to add the pretrained model's PCA (Principal Component Analysis) and I just threw in a confusion matrix to show that I know it exists (although it is interesting to see the model's most common misclassification is between glaciers and mountains)
- also a good place to add what images the classifiers (and really the fine-tuned models) struggled with
- the main problem was the building are usually on a street, and the correct label, for some images, could have been either building or street

# Decisions

## Optimizers
- I decided to use AdamW as the optimizer
- most of the literature and courses I've taken seem to default to Adam, but AdamW got me curious
- I read some posts and watched some education videos on the difference between the two optimizers and came to the conclusion that the difference is small, but AdamW is the mathematically correct optimizer, a corrected form of the original Adam optimizer
- a mathematically correct optimizer sounded more appealing to me, so I went with that
- I did try both and noted that AdamW took longer to train, about 1.5x
- I'm guessing that's because the less aggressive momentum dampening, but I could be wrong
- the accuracies differencies were negligable if anyone was curious

## ResNet-50
- early on in the project I decided to stick with one pre-trained model for the sklearn section
- although resnet-101 outperformeded in some cases, the comparison for the sklearn section was mainly between the classifiers themselves and not in search of the best accuracy
- I decided to stick to one set of hypers for all fine-tuning, I realize this makes the results not accurate, since every model likely had better hypers that would optimize their accuracies, however, that would be hundereds of runs and the point of the project would shift to just brute force grid search computing and I have other projects to invest time and compute into
- all that to say that resnet, although the winner on paper, could have been outperformed under certain hyper configs
- also to say that a fine-tuned resnet model would have likely had an even better final accuracy if I spent the time to optimize their hypers as well

## Staged Unfreezing
- I made the decision to try unfreezing certain layers after the models reached a plateau in val acc
- for 'staged unfreezing' (just what I called it), I unfroze the last three layers in sequence, and for 'full fine-tuning', I just unfroze the whole model with the last added layer
- this is important to note because if you search up 'staged fine-tuning' on the web, a paper from 2018 by Sebastian Ruder and Jeramy Howard detailing ULMFiT (Universal Language Model Fine-Tuning), in which they describe unfreezing a layer after each epoch
- since the models I was using didn't have that many layers (ULMFit was used on models with hundreds of layers), I modified the idea and just ran the model till early stopping, unfroze the last layer, and repeated a few times

# Not Shown

## Data Size
- I tried seeing the impact the data size would make on the model accuracies
- I decided not to include this data because the results were not very notable
- the models were all run with a forth, a half, and the full data set
- the models only crossed a notable threshold of 'learnt something' when the full dataset was applied
- one interesting thing I did note was this threshold, how at some point the models had enough data not to overfit and learned representations of the output labels well enough for the task

## Weight Decay
- I also tried varying the weight decay on the optimizer, from zero to 1e-4 to 1e-3
- the results were negligable, so I did not include them
- one thing to note though, was that there was no standard weight decay that the best across all models
- from model to model, and from run to run, the best weight decay jumped between zero and 1e-4

## Patience
- I applied early stopping with a patience value of 10 epochs, quite standard I'm sure, but just something to note
- why I think it's something to note is because I always wondered about the difference in patience numbers
- meaning is it just a backstop to confirm the val acc has plateaued or more of a hedge against the optimizer suddenly cresting the saddlebend of a n-dimensional loss curve and will suddenly learn more
- just some interesting thoughts I had that I wanted to share

# Further Research
- there are so many more hyperparameters to manipulate and tweek to try to achieve a slightly better model acc
- so many optimizer settings, different criterion, types and sizes of last layers, just endless
- I chose to focus on the ones that I was thought were the most impactful, i.e. learning rate, dropout, etc.
- so many classifier options to choose from and experiment with and tweek
- what I did with transfer learning was very rudimentary, meaning I took the best config for my scratch CNN and applied those hypers to the different models for training, which isn't optimal
- in an ideal world, every model would have there own grid search of hypers to optimize, so in reality, the acc numbers are quite meaningless as only to state which model for transfer learning is best with a certain static set of hypers
- nonetheless, still a worthwhile deep dive into the different available popular models and their implementations in the two most prominent ml architectures