Packages

```python
import numpy as np
from pydub import AudioSegment
import random
import sys
import io
import os
import glob
import IPython
from td_utils import *
%matplotlib inline
```

Data synthesis: Creating a Speech Dataset

- a speech dataset should ideally be as close as possible to the application it will be run on
- in this case, detecting the word ‘activate’ in working enviroments
- therefore, create recordings with a mix of positive words and negative words on different background sounds
- Listening to the Data
    - a friend has gone to libraries, cafes, restaurants, homes and offices all around the region to record background noises, as well as snippets of audio of people saying positive/negative words
    - the dataset includes a variety of accents
    - synthesize a dataset to train the model
        - the ‘activate’ directory contains positive examples of people saying the word ‘activate’
        - the ‘negatives’ directory contains negative examples of people saying random words other than ‘activate’
        - there is one word per audio recording
        - the ‘backgrounds’ directory contains ten second clips of background noise in different environments
- From Audio Recordings to Spectrograms
    - a microphone records little variations in air pressure over time, and it is these little variations in air pressure that the human ear perceives as sound
    - think of an audio recording as a long list of numbers measuring the little air pressure changes detected by the mic
    - use audio sampled at 44100 Hz
        - this means the mic gives 44,100 numbers per second
        - thus, a ten second audio clip is represented by 441,00 numbers
    - Spectrogram
        - it is quite difficult to figure out from this ‘raw’ representation of audio whether the word ‘activate’ was said
        - in order to help the sequence model more easily learn to detect trigger words, compute a spectrogram of the audio
        - the spectrogram describes how much different frequencies are present in an audio clip at any moment in time
        - Fourier transforms
            - a spectrogram is computed by sliding a window over the raw audio signal, and calculating the most active frequencies in each window using a Fourier transform

        ```python
        x = graph_spectrogram("audio_examples/example_train.wav")
        ```

    - the graph represents how active each frequency is over a number of time-steps

    - the color in the spectrogram shows the degree to which different frequencies are present (loud) in the audio at different points in time
    - green means a certain frequency is more active or more present in the audio clip (louder)
    - blue squares denote less active frequencies
    - the dimension of the output spectrogram depends upon the hyperparams of the spectrogram software and the length of the input
    - will be working with 10 sec audio clips as the standard length for the training examples
        - the number of timesteps of the spectrogram will be 5511
        - the spectrogram will be the input to the network, so Tx = 5511

    ```python
    _, data = wavfile.read("audio_examples/example_train.wav")
    print("Time steps in audio recording before spectrogram", data[:,0].shape)
    print("Time steps in input after spectrogram", x.shape)

    '''
    Time steps in audio recording before spectrogram (441000,)
    Time steps in input after spectrogram (101, 5511)
    '''
    ```

    ```python
    Tx = 5511 # The number of time steps input to the model from the spectrogram
    n_freq = 101 # Number of frequencies input to the model at each time step of the spectrogram
    ```

    - Dividing into time-intervals
        - a ten second interval of time can be divided with different units (steps)
        - raw audio divides ten second into 441,000 units
        - a spectrogram divides ten secs into 5,511 units
        - pydub Python module synthesizes audio, and it divides ten secs into 10k units
        - the output of the model will divide ten secs into 1,375 units
            - Ty = 1375
            - for each of the 1375 time steps, the model predicts whether someone recently finished saying the trigger word ‘activate’
        - all of these are hypers and can be changed (except the 441000, which is a function of the microphone)
        - chosen values that are within the standard range used for speech systems

        ```python
        Ty = 1375 # The number of time steps in the output of our model
        ```

- Generating a Single Training Example
    - Benefits of synthesizing data
        - because speech data is hard to acquire and label, synthesize the training data using the audio clips of activates, negatives, and backgrounds
        - it is quite slow to record lots of ten sec audio clips with random ‘activates’ in it
        - instead, it is easier to record lots of positives and negative words, and record background noise separately
    - Process for Synthesizing an audio clip
        - to synthesize a single training example,
            - pick a random ten sec background audio clip
            - randomly insert 0-4 audio clips of ‘activate’ into the ten sec clip
            - randomly insert 0-2 audio clips of negative words into the ten sec clip
        - because the word ‘activate’ is synthesized, the exact location of the word in known in the ten sec clip → easier to generate the labels y<t>
    - Pydub
        - use the pydub package to manipulate audio
        - pydub converts raw audio files into lists of pydub data structs
        - pydub uses 1ms as the discretization interval → hence 10k steps

    ```python
    # Load audio segments using pydub 
    activates, negatives, backgrounds = load_raw_audio('./raw_data/')

    print("background len should be 10,000, since it is a 10 sec clip\n" + str(len(backgrounds[0])),"\n")
    print("activate[0] len may be around 1000, since an `activate` audio clip is usually around 1 second (but varies a lot) \n" + str(len(activates[0])),"\n")
    print("activate[1] len: different `activate` clips can have different lengths\n" + str(len(activates[1])),"\n")

    '''
    background len should be 10,000, since it is a 10 sec clip
    10000 

    activate[0] len may be around 1000, since an `activate` audio clip is usually around 1 second (but varies a lot) 
    916 

    activate[1] len: different `activate` clips can have different lengths
    1579 
    '''
    ```

    - Overlaying positive/negative ‘word’ audio clips on top of the background audio
        - given a ten sec background clip and a short audio clip containing a positive or negative word, add the word audio clip on top of the background audio
        - since there will be multiple insertions into one clip, clips should not overlap
            - keep track of the times of previously inserted audio clips
    - Label the positive/negative words
        - recall that the labels y<t> represent whether or not someone has just finished saying activate
            - y<t> = 1 when that clip has finished saying ‘activate’
            - given a background clip, initialize y<t> = 0 for all t, since the clip doesn’t contain any ‘activate’
        - when inserting or overlaying an active clip, update labels for y<t>
            - rather than updating the label of a single time step, update 50 steps oft eh output to have target label 1
        - train a GRU (Gated Recurrent Unit) to detect when someone has finished saying ‘activate’
    - Example
        - suppose the synthesized ‘activate’ clip ends at the five second mark in the ten second audio
        - recall that Ty = 1375, so timestep 687 = int(1375*0.5) corresponds to the moment five seconds into the audio clip
        - Set y<688> = 1
        - allow the GRU to detect ‘activate’ anywhere within a short time-interval after this moment, so actually set 50 consecutive values of the label y<t> to 1
    - Synthesized data is easier to label
        - this is another reason for synthesizing the training data; it’s relatively straightforward to generate these labels y<t> as described above
        - in contrast, the 10 sec of audio recorded on a mic, it’s quite time consuming for a person to listen to it and mark manually exactly when ‘activate’ finished
    - Visualizing the labels
        - a figure illustrating the labels y<t> in a clip
        - inserted ‘activate’, ‘innocent’, ‘activate’, ‘baby’
        - note that the positive labels 1 are associated only with the positive words

    - Helper functions
        - to implement the training set synthesis process, below are helper functions
        - all of these function will use a 1ms discretization interval
        - the ten seconds of audio is always discretized into 10k steps
            - get_random_time_segment(segement_ms): retrieves a random time segment from the background audio
            - is_overlapping(segment_time, existing_segments): checks if a time segment overlaps with existing segements
            - insert_audio_clip(background, audio_clip, existing_times): inserts an audio segment at a random time int he background audio and uses the function above
            - insert_ones(y, segement_end_ms)
    - Get a random time segment
        - the function get_random_time_segement(segment_ms) returns a random time segment onto which can be inserted an audio clip of duration segment_ms

        ```python
        def get_random_time_segment(segment_ms):
            """
            Gets a random time segment of duration segment_ms in a 10,000 ms audio clip.

            Arguments:
            segment_ms -- the duration of the audio clip in ms ("ms" stands for "milliseconds")

            Returns:
            segment_time -- a tuple of (segment_start, segment_end) in ms
            """

            segment_start = np.random.randint(low=0, high=10000-segment_ms)   # Make sure segment doesn't run past the 10sec background 
            segment_end = segment_start + segment_ms - 1

            return (segment_start, segment_end)
        ```

    - Check if audio clips are overlapping
        - suppose audio clips are inserted at (1000, 1800), and (3400, 4500)
            - the first segment starts at step 1000 and ends at step 1800
            - the second segment starts at 3400 and ends at 4500
        - considering whether to insert a new audio clip at (3000, 3600), does this overlap with one of the previously inserted segments?
            - in this case, they do
        - for the purpose of this function, define (100, 200) and (200, 250) to be overlapping, since they overlap at timestep 200
    - Exercise 1 - is_overlapping
        - implement the function to check if a new time segment overlaps with any of the previous segments
        - create a ‘false’ flag, that will be set to true if there is an overlap
        - loop over the previous_segments’ start and end times
        - compare these times to the segment’s start and end times
        - if there is an overlap, set the flag to True

        ```python
        # UNQ_C1
        # GRADED FUNCTION: is_overlapping

        def is_overlapping(segment_time, previous_segments):
            """
            Checks if the time of a segment overlaps with the times of existing segments.

            Arguments:
            segment_time -- a tuple of (segment_start, segment_end) for the new segment
            previous_segments -- a list of tuples of (segment_start, segment_end) for the existing segments

            Returns:
            True if the time segment overlaps with any of the existing segments, False otherwise
            """

            segment_start, segment_end = segment_time

            ### START CODE HERE ### (≈ 4 lines)
            # Step 1: Initialize overlap as a "False" flag. (≈ 1 line)
            overlap = False

            # Step 2: loop over the previous_segments start and end times.
            # Compare start/end times and set the flag to True if there is an overlap (≈ 3 lines)
            for previous_start, previous_end in previous_segments:
                if segment_start  <= previous_end and segment_end >= previous_start:
                    overlap = True
                    break
            ### END CODE HERE ###

            return overlap
        ```

        ```python
        overlap1 = is_overlapping((950, 1430), [(2000, 2550), (260, 949)])
        overlap2 = is_overlapping((2305, 2950), [(824, 1532), (1900, 2305), (3424, 3656)])
        print("Overlap 1 = ", overlap1)
        print("Overlap 2 = ", overlap2)

        '''
        Overlap 1 =  False
        Overlap 2 =  True
        '''
        ```

    - Insert audio clip
        - use the previous helper function to insert a new audio clip onto the ten sec background at a random time
        - ensure that any newly inserted segment doesn’t overlap with previously inserted segments
    - Exercise 2 - insert_audio_clip
        - overlay an audio clip onto the background 10 sec clip
        - get the length of the audio clip that is to be inserted
            - get a random time segment whose duration equals the duration of the audio clip that is to be inserted
        - make sure that the time segment does not overlap with any of the previous time segments
            - if it is overlapping, then go back to the first step and pick a new time segment
        - append the new time segment tot he list of existing time segments
            - this keeps track of all the segments inserted
        - overlay the audio clip over the background using pydub

        ```python
        # UNQ_C2
        # GRADED FUNCTION: insert_audio_clip

        def insert_audio_clip(background, audio_clip, previous_segments):
            """
            Insert a new audio segment over the background noise at a random time step, ensuring that the 
            audio segment does not overlap with existing segments.

            Arguments:
            background -- a 10 second background audio recording.  
            audio_clip -- the audio clip to be inserted/overlaid. 
            previous_segments -- times where audio segments have already been placed

            Returns:
            new_background -- the updated background audio
            """

            # Get the duration of the audio clip in ms
            segment_ms = len(audio_clip)

            ### START CODE HERE ### 
            # Step 1: Use one of the helper functions to pick a random time segment onto which to insert 
            # the new audio clip. (≈ 1 line)
            segment_time = get_random_time_segment(segment_ms)

            # Step 2: Check if the new segment_time overlaps with one of the previous_segments. If so, keep 
            # picking new segment_time at random until it doesn't overlap. To avoid an endless loop
            # we retry 5 times(≈ 2 lines)
            retry = 5 
            while is_overlapping(segment_time, previous_segments) and retry >= 0:
                segment_time = get_random_time_segment(segment_ms)
                retry = retry - 1
            ### END CODE HERE ###
                #print(segment_time)
            # if last try is not overlaping, insert it to the background
            if not is_overlapping(segment_time, previous_segments):
            ### START CODE HERE ### 
                # Step 3: Append the new segment_time to the list of previous_segments (≈ 1 line)
                previous_segments.append(segment_time)
            ### END CODE HERE ###
                # Step 4: Superpose audio segment and background
                new_background = background.overlay(audio_clip, position = segment_time[0])
            else:
                #print("Timeouted")
                new_background = background
                segment_time = (10000, 10000)

            return new_background, segment_time
        ```

        ```python
        np.random.seed(5)
        audio_clip, segment_time = insert_audio_clip(backgrounds[0], activates[0], [(3790, 4400)])
        audio_clip.export("insert_test.wav", format="wav")
        print("Segment Time: ", segment_time)
        IPython.display.Audio("insert_test.wav")

        # Segment Time:  (2254, 3169)
        ```

    - Insert ones for the labels of the positive target
        - implement code to update the labels y<t>, assuming an ‘activate’ audio clip was just inserted
        - y is a (1, 1375) dimensional vector, since Ty = 1375
        - if the ‘activate’ audio clip ends at time step t, then set y<t+1> = 1 and also set the next 49 additional consecutive values to 1
            - notice that if the target word appears near the end of the entire audio clip, there may not be 50 additional time steps to set to 1
            - make sure to not run off the end of the array and try to update y[0][1375], since the valid indices are y[0][0] through y[0][1375] because Ty = 1375
            - so if ‘activate’ ends at step 1370, only set y[0][1371] = y[0][1372] = y[0][1373] = y[0][1374] = 1
    - Exercise 3 - insert_ones
        - use a for loop
        - can use python array slicing operations
        - if a segment ends at segement_end_ms, convert it to the indexing for the outputs y (using a 1375 step discretization)

        ```python
        segment_end_y = int(segment_end_ms * Ty / 10000.0)
        ```

        ```python
        # UNQ_C3
        # GRADED FUNCTION: insert_ones

        def insert_ones(y, segment_end_ms):
            """
            Update the label vector y. The labels of the 50 output steps strictly after the end of the segment 
            should be set to 1. By strictly we mean that the label of segment_end_y should be 0 while, the
            50 following labels should be ones.

            Arguments:
            y -- numpy array of shape (1, Ty), the labels of the training example
            segment_end_ms -- the end time of the segment in ms

            Returns:
            y -- updated labels
            """
            _, Ty = y.shape

            # duration of the background (in terms of spectrogram time-steps)
            segment_end_y = int(segment_end_ms * Ty / 10000.0)

            if segment_end_y < Ty:
                # Add 1 to the correct index in the background label (y)
                ### START CODE HERE ### (≈ 3 lines)
                for i in range(segment_end_y + 1, segment_end_y + 51):
                    if i < Ty:
                        y[0, i] = 1
                ### END CODE HERE ###

            return y
        ```

        ```python
        arr1 = insert_ones(np.zeros((1, Ty)), 9700)
        plt.plot(insert_ones(arr1, 4251)[0,:])
        print("sanity checks:", arr1[0][1333], arr1[0][634], arr1[0][635])
        ```

    - Creating a training example
        - use insert_audio_clip and insert_ones to create a new training example
    - Exercise 4 - create_training_example
        - initialize the label vector y as a numpy array of zeros and shape (1, Ty)
        - initialize the set of existing segments to an empty list
        - randomly select 0 to 4 ‘activate’ audio clips, and insert them onto the ten sec clip, also insert labels at the correct position in the label vector y
        - randomly select 0 to 2 negative audio clips, and insert them into the ten sec clip

        ```python
        # UNQ_C4
        # GRADED FUNCTION: create_training_example

        def create_training_example(background, activates, negatives, Ty):
            """
            Creates a training example with a given background, activates, and negatives.

            Arguments:
            background -- a 10 second background audio recording
            activates -- a list of audio segments of the word "activate"
            negatives -- a list of audio segments of random words that are not "activate"
            Ty -- The number of time steps in the output

            Returns:
            x -- the spectrogram of the training example
            y -- the label at each time step of the spectrogram
            """

            # Make background quieter
            background = background - 20

            ### START CODE HERE ###
            # Step 1: Initialize y (label vector) of zeros (≈ 1 line)
            y = np.zeros((1, Ty))

            # Step 2: Initialize segment times as empty list (≈ 1 line)
            previous_segments = []
            ### END CODE HERE ###

            # Select 0-4 random "activate" audio clips from the entire list of "activates" recordings
            number_of_activates = np.random.randint(0, 5)
            random_indices = np.random.randint(len(activates), size=number_of_activates)
            random_activates = [activates[i] for i in random_indices]

            ### START CODE HERE ### (≈ 3 lines)
            # Step 3: Loop over randomly selected "activate" clips and insert in background
            for one_random_activate in random_activates:
                # Insert the audio clip on the background
                background, segment_time = insert_audio_clip(background, one_random_activate, previous_segments)
                # Retrieve segment_start and segment_end from segment_time
                segment_start, segment_end = segment_time
                # Insert labels in "y" at segment_end
                y = insert_ones(y, segment_end)
            ### END CODE HERE ###

            # Select 0-2 random negatives audio recordings from the entire list of "negatives" recordings
            number_of_negatives = np.random.randint(0, 3)
            random_indices = np.random.randint(len(negatives), size=number_of_negatives)
            random_negatives = [negatives[i] for i in random_indices]

            ### START CODE HERE ### (≈ 2 lines)
            # Step 4: Loop over randomly selected negative clips and insert in background
            for random_negative in random_negatives:
                # Insert the audio clip on the background 
                background, _ = insert_audio_clip(background, random_negative, previous_segments)
            ### END CODE HERE ###

            # Standardize the volume of the audio clip 
            background = match_target_amplitude(background, -20.0)

            # Export new training example 
            file_handle = background.export("train" + ".wav", format="wav")

            # Get and plot spectrogram of the new recording (background with superposition of positive and negatives)
            x = graph_spectrogram("train.wav")

            return x, y
        ```

        ```python
        # Set the random seed
        np.random.seed(18)
        x, y = create_training_example(backgrounds[0], activates, negatives, Ty)
        ```

        ```python
        plt.plot(y[0])
        ```

- Full Training Set
    - use this process to generate a large training set

    ```python
    # START SKIP FOR GRADING
    np.random.seed(4543)
    nsamples = 32
    X = []
    Y = []
    for i in range(0, nsamples):
        if i%10 == 0:
            print(i)
        x, y = create_training_example(backgrounds[i % 2], activates, negatives, Ty)
        X.append(x.swapaxes(0,1))
        Y.append(y.swapaxes(0,1))
    X = np.array(X)
    Y = np.array(Y)
    # END SKIP FOR GRADING
    ```

    - to save the dataset into a file to be loaded later

        ```python
        # Save the data for further uses
        # np.save(f'./XY_train/X.npy', X)
        # np.save(f'./XY_train/Y.npy', Y)
        # Load the preprocessed training examples
        # X = np.load("./XY_train/X.npy")
        # Y = np.load("./XY_train/Y.npy")
        ```

- Development Set
    - to test the model, dev set was recorded of 25 examples
    - while the training data is synthesized, create a dev set using the same distribution as the real inputs
    - thus, record 25 10sec audio clips of people saying ‘activate’ and other random words, and label them by hand
    - this follows the principle that the dev set should be as similar as possible tot he test set distribution

    ```python
    # Load preprocessed dev set examples
    X_dev = np.load("./XY_dev/X_dev.npy")
    Y_dev = np.load("./XY_dev/Y_dev.npy")
    ```

The Model

- write and train a trigger word detection model
- the model will use 1D conv layers, GRU layers, and dense layers
- load the packages that will allow the use of layers in Keras

```python
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from tensorflow.keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from tensorflow.keras.optimizers import Adam
```

- Build the Model
    - the goal is to build a network that will ingest a spectrogram and output a signal when it detects the trigger word
    - four layers
        - a conv layer
        - two GRU layers
        - a dense layer

    - 1D conv layer
        - it inputs the 5511 step spectogram
        - each step is a vector of 101 units
        - it outputs a 1375 step output
        - the output is further processed by multiple layers to get the final Ty = 1375 step output
        - the 1D conv layer plays a role similar to the 2D conv of extracting low-level features and then possibly generating an output of smaller dimension
        - computationally, the 1D conv layer also helps speed up the model because now the GRU can process only 1375 timesteps rather than 5511 timesteps
    - GRU, dense and sigmoid
        - the two GRU layers read the sequence of inputs from left to right
        - a dense plus sigmoid layer makes a prediction for y<t>
        - because y is a binary value, use a sigmoid output at the last layer to estimate the chance of the output being 1, corresponding to the user having just said ‘activate’
    - Unidirectional RNN
        - this is because focusing on past context is more critical than future context for detecting the activation word
        - reduces computational overhead, as the model does not process information in both directions
        - in real application, trigger word detection could be performed every second using a sliding ten sec window
        - therefore, even if the activation word could appear at any point within these ten sec, it can be detected immediately upon being spoken
        - if the alg initially fails to detect it, further detections with each subsequent second provide additional opportunities for recognition
        - this approach efficiently balances performance with timely detection attempts
    - Implement the model
        - the input of each layer is the output of the previous one
        - Step 1: CONV layer, use Conv1D() with 196 filters, a filter size of 15, and stride of 4

            ```python
            output_x = Conv1D(filters=...,kernel_size=...,strides=...)(input_x)
            ```

            - followed by batch normalization

                ```python
                output_x = BatchNormalization()(input_x)
                ```

            - followed by a ReLu activation

                ```python
                output_x = Activation("...")(input_x)
                ```

            - followed by dropout with rate of 0.8

                ```python
                output_x = Dropout(rate=...)(input_x)
                ```

        - Step 2: first GRU layer, use 128 units

            ```python
            output_x = GRU(units=..., return_sequences = ...)(input_x)
            ```

            - return sequences instead of just the last time step’s prediction to ensure that all the GRU’s hidden states are fed to the next layer
            - follow with dropout, rate of 0.8
            - followed with batch norm

                ```python
                output_x = BatchNormalization()(input_x)
                ```

        - Step 3: second GRU layer, same specifications
            - follow with dropout, batch norm, and another dropout
        - Step 4: create a time-distributed dense layer

            ```python
            output_x = TimeDistributed(Dense(1, activation = "sigmoid"))(input_x)
            ```

            - this creates a dense layer followed by a sigmoid, so that the parameters used for the dense layer are the same for every time step
    - Exercise 5 - modelf

        ```python
        # UNQ_C5
        # GRADED FUNCTION: modelf

        def modelf(input_shape):
            """
            Function creating the model's graph in Keras.

            Argument:
            input_shape -- shape of the model's input data (using Keras conventions)

            Returns:
            model -- Keras model instance
            """

            X_input = Input(shape = input_shape)

            ### START CODE HERE ###

            # Step 1: CONV layer (≈4 lines)
            # Add a Conv1D with 196 units, kernel size of 15 and stride of 4
            X = Conv1D(filters=196, kernel_size=15, strides=4)(X_input)
            # Batch normalization
            X = BatchNormalization()(X)
            # ReLu activation
            X = Activation('relu')(X)
            # dropout (use 0.8)
            X = Dropout(rate=0.8)(X)                  

            # Step 2: First GRU Layer (≈4 lines)
            # GRU (use 128 units and return the sequences)
            X = GRU(units=128, return_sequences=True)(X)
            # dropout (use 0.8)
            X = Dropout(0.8)(X)
            # Batch normalization.
            X = BatchNormalization()(X)                 

            # Step 3: Second GRU Layer (≈4 lines)
            # GRU (use 128 units and return the sequences)
            X = GRU(units=128, return_sequences=True)(X)
            # dropout (use 0.8)
            X = Dropout(0.8)(X)
            # Batch normalization
            X = BatchNormalization()(X)
            # dropout (use 0.8)
            X = Dropout(rate=0.8)(X)               

            # Step 4: Time-distributed dense layer (≈1 line)
            # TimeDistributed  with sigmoid activation 
            X = TimeDistributed(Dense(1, activation='sigmoid'))(X)

            ### END CODE HERE ###

            model = Model(inputs = X_input, outputs = X)

            return model  
        ```

        ```python
        model = modelf(input_shape = (Tx, n_freq))
        ```

        ```python
        model.summary()

        '''
        Model: "functional_3"
        _________________________________________________________________
        Layer (type)                 Output Shape              Param #   
        =================================================================
        input_4 (InputLayer)         [(None, 5511, 101)]       0         
        _________________________________________________________________
        conv1d_2 (Conv1D)            (None, 1375, 196)         297136    
        _________________________________________________________________
        batch_normalization_4 (Batch (None, 1375, 196)         784       
        _________________________________________________________________
        activation_2 (Activation)    (None, 1375, 196)         0         
        _________________________________________________________________
        dropout_4 (Dropout)          (None, 1375, 196)         0         
        _________________________________________________________________
        gru_2 (GRU)                  (None, 1375, 128)         125184    
        _________________________________________________________________
        dropout_5 (Dropout)          (None, 1375, 128)         0         
        _________________________________________________________________
        batch_normalization_5 (Batch (None, 1375, 128)         512       
        _________________________________________________________________
        gru_3 (GRU)                  (None, 1375, 128)         99072     
        _________________________________________________________________
        dropout_6 (Dropout)          (None, 1375, 128)         0         
        _________________________________________________________________
        batch_normalization_6 (Batch (None, 1375, 128)         512       
        _________________________________________________________________
        dropout_7 (Dropout)          (None, 1375, 128)         0         
        _________________________________________________________________
        time_distributed_1 (TimeDist (None, 1375, 1)           129       
        =================================================================
        Total params: 523,329
        Trainable params: 522,425
        Non-trainable params: 904
        _________________________________________________________________
        '''
        ```

        - the output of the network is of shape (None, 1385, 1) while the input is (None, 5511, 101)
        - the Conv1D has reduced the number of steps from 551 to 1375
    - Fit the Model
        - trigger word detection takes a long time to train
        - to save time, the model was pretrained for three hours with a training set of around 4k examples

        ```python
        from tensorflow.keras.models import model_from_json

        json_file = open('./models/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights('./models/model.h5')
        ```

        - Block Training for BatchNormalization layers
            - when fine-tuning a pretrained model, it is important to block the weights of all the batchnormalization layers

                ```python
                model.layers[2].trainable = False
                model.layers[7].trainable = False
                model.layers[10].trainable = False
                ```

            - train the model further

                ```python
                opt = Adam(lr=1e-6, beta_1=0.9, beta_2=0.999)
                model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
                ```

                ```python
                model.fit(X, Y, batch_size = 16, epochs=1)

                # 2/2 [==============================] - 3s 2s/step - loss: 0.1688 - accuracy: 0.9462
                ```

    - Test the Model
        - see how the model performs on the dev set

            ```python
            loss, acc, = model.evaluate(X_dev, Y_dev)
            print("Dev set accuracy = ", acc)

            '''
            1/1 [==============================] - 0s 1ms/step - loss: 0.1887 - accuracy: 0.9239
            Dev set accuracy =  0.9238981604576111
            '''
            ```

        - the accuracy isn’t a great metric for this tast
        - since the labels are heavily skewed to 0’s, a NN that just outputs 0’s would get slight over 90% accuracy
        - define a more useful metric such as F1 score or Precision/Recall

Making Predictions

- use the model to make predictions

    ```python
    def detect_triggerword(filename):
        plt.subplot(2, 1, 1)

        # Correct the amplitude of the input file before prediction 
        audio_clip = AudioSegment.from_wav(filename)
        audio_clip = match_target_amplitude(audio_clip, -20.0)
        file_handle = audio_clip.export("tmp.wav", format="wav")
        filename = "tmp.wav"

        x = graph_spectrogram(filename)
        # the spectrogram outputs (freqs, Tx) and we want (Tx, freqs) to input into the model
        x  = x.swapaxes(0,1)
        x = np.expand_dims(x, axis=0)
        predictions = model.predict(x)

        plt.subplot(2, 1, 2)
        plt.plot(predictions[0,:,0])
        plt.ylabel('probability')
        plt.show()
        return predictions
    ```

- Insert a chime to acknowledge the ‘activate’ trigger
    - once the probability of having detected the word ‘activate’ at each output step has been estimated, trigger a ‘chiming’ sound to play when the probability is above a certain threshold
    - y<t> might be near 1 for many values in a row after ‘activate’ is said, yet only one chime is needed
        - so insert chime sound at most once every 75 steps
        - this will help prevent from inserting two chimes for a single instance of ‘activate’
        - this player a role similar to non-max suppression from CV

        ```python
        chime_file = "audio_examples/chime.wav"
        def chime_on_activate(filename, predictions, threshold):
            audio_clip = AudioSegment.from_wav(filename)
            chime = AudioSegment.from_wav(chime_file)
            Ty = predictions.shape[1]
            # Step 1: Initialize the number of consecutive output steps to 0
            consecutive_timesteps = 0
            i = 0
            # Step 2: Loop over the output steps in the y
            while i < Ty:
                # Step 3: Increment consecutive output steps
                consecutive_timesteps += 1
                # Step 4: If prediction is higher than the threshold for 20 consecutive output steps have passed
                if consecutive_timesteps > 20:
                    # Step 5: Superpose audio and background using pydub
                    audio_clip = audio_clip.overlay(chime, position = ((i / Ty) * audio_clip.duration_seconds) * 1000)
                    # Step 6: Reset consecutive output steps to 0
                    consecutive_timesteps = 0
                    i = 75 * (i // 75 + 1)
                    continue
                # if amplitude is smaller than the threshold reset the consecutive_timesteps counter
                if predictions[0, i, 0] < threshold:
                    consecutive_timesteps = 0
                i += 1

            audio_clip.export("chime_output.wav", format='wav')
        ```

- Test on Dev Examples
    - how does the model perform on two unseen audio clips from the dev set

        ```python
        filename = "./raw_data/dev/1.wav"
        prediction = detect_triggerword(filename)
        chime_on_activate(filename, prediction, 0.5)
        IPython.display.Audio("./chime_output.wav")
        ```

        ```python
        filename  = "./raw_data/dev/2.wav"
        prediction = detect_triggerword(filename)
        chime_on_activate(filename, prediction, 0.5)
        IPython.display.Audio("./chime_output.wav")
        ```

Try Your Own Example

- trim audio to 10 sec max or pad to 10 sec

```python
# Preprocess the audio to the correct format
def preprocess_audio(filename):
    # Trim or pad audio segment to 10000ms
    padding = AudioSegment.silent(duration=10000)
    segment = AudioSegment.from_wav(filename)[:10000]
    segment = padding.overlay(segment)
    # Set frame rate to 44100
    segment = segment.set_frame_rate(44100)
    # Export as wav
    segment.export(filename, format='wav')
```

```python
your_filename = "audio_examples/my_audio.wav"

preprocess_audio(your_filename)
IPython.display.Audio(your_filename) # listen to the audio you uploaded 
```

```python
chime_threshold = 0.8
prediction = detect_triggerword(your_filename)
chime_on_activate(your_filename, prediction, chime_threshold)
IPython.display.Audio("./chime_output.wav")
```