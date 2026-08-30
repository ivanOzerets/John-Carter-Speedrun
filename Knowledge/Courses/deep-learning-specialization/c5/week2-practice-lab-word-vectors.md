Packages

```python
import numpy as np
from w2v_utils import *
```

Load the Word Vectors

- use 50D GloVe vectors to represent words

    ```python
    words, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')
    ```

- loaded:
    - word: set of words in the vocabulary
    - word_to_vec_map: dictionary mapping words to their GloVe vector representation

Embedding Vectors Versus One-Hot Vectors

- one-hot vectors don’t do a good  job of capturing the level of similarity between words
- embedding vectors, such as GloVe vectors, provide much more useful information about the meaning of individual words

Cosine Similarity

- to measure the similarity between two words, need a way to measure the degree of similarity between two embedding vectors for the two words
- given two vectors u and v, cosine similarity is defined as follows

- u . v is the dot product of two vectors
- ||u||2 is the norm of the vector
- theta is the angle between u and v
- the cosine similarity depends on the angle between u and v
    - if u and v are very similar, their cosine similarity will take a smaller value
    - if they are dissimilar, the cosine similarity will take a smaller value

- Exercise 1 - cosine_similarity
    - implement the function cosine_similarity() to evaluate the similarity between word vectors
    - the norm of u is defined as

    ```python
    # UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
    # GRADED FUNCTION: cosine_similarity

    def cosine_similarity(u, v):
        """
        Cosine similarity reflects the degree of similarity between u and v

        Arguments:
            u -- a word vector of shape (n,)          
            v -- a word vector of shape (n,)

        Returns:
            cosine_similarity -- the cosine similarity between u and v defined by the formula above.
        """

        # Special case. Consider the case u = [0, 0], v=[0, 0]
        if np.all(u == v):
            return 1

        ### START CODE HERE ###
        # Compute the dot product between u and v (≈1 line)
        dot = u @ v 
        # Compute the L2 norm of u (≈1 line)
        norm_u = np.sqrt(np.sum(u ** 2))

        # Compute the L2 norm of v (≈1 line)
        norm_v = np.sqrt(np.sum(v ** 2))

        # Avoid division by 0
        if np.isclose(norm_u * norm_v, 0, atol=1e-32):
            return 0

        # Compute the cosine similarity defined by formula (1) (≈1 line)
        cosine_similarity = dot / (norm_u * norm_v)
        ### END CODE HERE ###

        return cosine_similarity

    '''
    cosine_similarity(father, mother) =  0.8909038442893615
    cosine_similarity(ball, crocodile) =  0.2743924626137942
    cosine_similarity(france - paris, rome - italy) =  -0.6751479308174201
    '''
    ```

Word Analogy Task

- in a word analogy task, complete this sentence: “a is to b as c is to ___”
- trying to find a word d, such that the associated word vectors ea, eb, ec, ed are related int he following manner: eb - ea ~= ed - ec
- measure the similarity between eb - ea and ed - ec using cosine similarity
- Exercise 2 - complete_analogy

    ```python
    # UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
    # GRADED FUNCTION: complete_analogy

    def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
        """
        Performs the word analogy task as explained above: a is to b as c is to ____. 

        Arguments:
        word_a -- a word, string
        word_b -- a word, string
        word_c -- a word, string
        word_to_vec_map -- dictionary that maps words to their corresponding vectors. 

        Returns:
        best_word --  the word such that v_b - v_a is close to v_best_word - v_c, as measured by cosine similarity
        """

        # convert words to lowercase
        word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()

        ### START CODE HERE ###
        # Get the word embeddings e_a, e_b and e_c (≈1-3 lines)
        e_a, e_b, e_c = [word_to_vec_map[word] for word in [word_a, word_b, word_c]]
        ### END CODE HERE ###

        words = word_to_vec_map.keys()
        max_cosine_sim = -100              # Initialize max_cosine_sim to a large negative number
        best_word = None                   # Initialize best_word with None, it will help keep track of the word to output

        # loop over the whole word vector set
        for w in words:   
            # to avoid best_word being one the input words, skip the input word_c
            # skip word_c from query
            if w == word_c:
                continue

            ### START CODE HERE ###
            # Compute cosine similarity between the vector (e_b - e_a) and the vector ((w's vector representation) - e_c)  (≈1 line)
            cosine_sim = cosine_similarity(np.subtract(e_b, e_a), np.subtract(word_to_vec_map[w], e_c))

            # If the cosine_sim is more than the max_cosine_sim seen so far,
                # then: set the new max_cosine_sim to the current cosine_sim and the best_word to the current word (≈3 lines)
            if cosine_sim > max_cosine_sim:
                max_cosine_sim = cosine_sim
                best_word = w
            ### END CODE HERE ###

        return best_word
    ```

- test the code

    ```python
    # START SKIP FOR GRADING
    triads_to_try = [('italy', 'italian', 'spain'), ('india', 'delhi', 'japan'), ('man', 'woman', 'boy'), ('small', 'smaller', 'large')]
    for triad in triads_to_try:
        print ('{} -> {} :: {} -> {}'.format( *triad, complete_analogy(*triad, word_to_vec_map)))

    # END SKIP FOR GRADING

    '''
    italy -> italian :: spain -> spanish
    india -> delhi :: japan -> tokyo
    man -> woman :: boy -> girl
    small -> smaller :: large -> smaller
    '''
    ```

Debiasing Word Vectors

- examine gender biases that can be reflected in a word embedding, and explore algs for reducing the bias
- compute a vector g = e_woman - e_man, where e_woman represents the word vector corresponding tot he word woman, and e_man corresponds to the word vector corresponding to the word man
- the resulting vector g roughly encodes the concept of gender
- more accurate representation with g1 = e_mother - e_father, g2 = e_girl - e_boy, … and average over them, but just using one will give good enough results for now

    ```python
    g = word_to_vec_map['woman'] - word_to_vec_map['man']
    print(g)

    '''
    [-0.087144    0.2182     -0.40986    -0.03922    -0.1032      0.94165
     -0.06042     0.32988     0.46144    -0.35962     0.31102    -0.86824
      0.96006     0.01073     0.24337     0.08193    -1.02722    -0.21122
      0.695044   -0.00222     0.29106     0.5053     -0.099454    0.40445
      0.30181     0.1355     -0.0606     -0.07131    -0.19245    -0.06115
     -0.3204      0.07165    -0.13337    -0.25068714 -0.14293    -0.224957
     -0.149       0.048882    0.12191    -0.27362    -0.165476   -0.20426
      0.54376    -0.271425   -0.10245    -0.32108     0.2516     -0.33455
     -0.04371     0.01258   ]
    '''
    ```

- consider the cosine similarity of different words with g
- what does a positive value of similarity mean, versus a negative cosine similarity

    ```python
    print ('List of names and their similarities with constructed vector:')

    # girls and boys name
    name_list = ['john', 'marie', 'sophie', 'ronaldo', 'priya', 'rahul', 'danielle', 'reza', 'katy', 'yasmin']

    for w in name_list:
        print (w, cosine_similarity(word_to_vec_map[w], g))

    '''
    List of names and their similarities with constructed vector:
    john -0.23163356145973724
    marie 0.315597935396073
    sophie 0.3186878985941878
    ronaldo -0.31244796850329437
    priya 0.17632041839009402
    rahul -0.16915471039231722
    danielle 0.24393299216283895
    reza -0.07930429672199553
    katy 0.2831068659572615
    yasmin 0.23313857767928753
    '''
    ```

- female first names tend to have positive cosine similarity with the constructive vector g, while male first names tend to have a negative cosine similarity
- other words

    ```python
    print('Other words and their similarities:')
    word_list = ['lipstick', 'guns', 'science', 'arts', 'literature', 'warrior','doctor', 'tree', 'receptionist', 
                 'technology',  'fashion', 'teacher', 'engineer', 'pilot', 'computer', 'singer']
    for w in word_list:
        print (w, cosine_similarity(word_to_vec_map[w], g))

    '''
    Other words and their similarities:
    lipstick 0.27691916256382665
    guns -0.1888485567898898
    science -0.06082906540929699
    arts 0.008189312385880344
    literature 0.0647250443345993
    warrior -0.20920164641125288
    doctor 0.11895289410935045
    tree -0.07089399175478092
    receptionist 0.3307794175059374
    technology -0.13193732447554293
    fashion 0.035638946257727
    teacher 0.1792092343182567
    engineer -0.08039280494524072
    pilot 0.0010764498991917074
    computer -0.10330358873850498
    singer 0.18500518136496297
    '''
    ```

- here the results reflect certain unhealthy gender stereotypes
- some word pairs should remain gender specific, while other words should be neutralized
- Neutralize Bias for Non-Gender Specific Words
    - using a 50D word embedding, the 50D space can be split into two parts: the bias-direction g, and the remaining 49D, which is called g_orth
    - in linear algebra, the 49D is perpendicular to g, meaning it is at 90 degrees to g
    - the neutralization step takes a vector such as e_receptionist and zeros the component in the direction of g, giving e^debiased_receptionist

    - Exercise 3 - neutralize
        - remove the bias of words such as receptionist or scientist
        - given an input embedding e, use the following formulas to compute e^debiased

        - e^bias_component is the projection of e onto the direction g

        ```python
        # The paper assumes all word vectors to have L2 norm as 1 and hence the need for this calculation
        from tqdm import tqdm
        word_to_vec_map_unit_vectors = {
            word: embedding / np.linalg.norm(embedding)
            for word, embedding in tqdm(word_to_vec_map.items())
        }
        g_unit = word_to_vec_map_unit_vectors['woman'] - word_to_vec_map_unit_vectors['man']
        ```

        ```python
        def neutralize(word, g, word_to_vec_map):
            """
            Removes the bias of "word" by projecting it on the space orthogonal to the bias axis. 
            This function ensures that gender neutral words are zero in the gender subspace.

            Arguments:
                word -- string indicating the word to debias
                g -- numpy-array of shape (50,), corresponding to the bias axis (such as gender)
                word_to_vec_map -- dictionary mapping words to their corresponding vectors.

            Returns:
                e_debiased -- neutralized word vector representation of the input "word"
            """

            ### START CODE HERE ###
            # Select word vector representation of "word". Use word_to_vec_map. (≈ 1 line)
            e = word_to_vec_map[word]

            # Compute e_biascomponent using the formula given above. (≈ 1 line)
            e_biascomponent = ((e @ g) / np.linalg.norm(g)**2) * g

            # Neutralize e by subtracting e_biascomponent from it 
            # e_debiased should be equal to its orthogonal projection. (≈ 1 line)
            e_debiased = e - e_biascomponent
            ### END CODE HERE ###

            return e_debiased

        '''
        cosine similarity between receptionist and g, before neutralizing:  0.3307794175059374
        cosine similarity between receptionist and g_unit, after neutralizing:  3.5723165491646677e-17
        '''
        ```

- Equalization Alg for Gender-Specific Words
    - see how debiasing can also be applied to word pairs such actress and actor
    - equalization is applied to pairs of words that might want to have differ only through the gender property
    - suppose that actress is closer to babysit than actor
    - by applying neutralization to babysit, you can reduce the gender stereotype associated with babysitting
    - but this still does not guarantee that actor and actress are equidistant from babysit
    - the equalization alg takes are of this
    - the key idea behind equalization is to make sure that a particular pair of words are equidistant from the 49D g_ortho
    - the equalization step also ensures that the two equalized step are now the same distance from e^debiased_receptionist, or from any other work that has been neutralized

    - the derivation of the linear algebra

    - Exercise 4 - equlize

        ```python
        def equalize(pair, bias_axis, word_to_vec_map):
            """
            Debias gender specific words by following the equalize method described in the figure above.

            Arguments:
            pair -- pair of strings of gender specific words to debias, e.g. ("actress", "actor") 
            bias_axis -- numpy-array of shape (50,), vector corresponding to the bias axis, e.g. gender
            word_to_vec_map -- dictionary mapping words to their corresponding vectors

            Returns
            e_1 -- word vector corresponding to the first word
            e_2 -- word vector corresponding to the second word
            """

            ### START CODE HERE ###
            # Step 1: Select word vector representation of "word". Use word_to_vec_map. (≈ 2 lines)
            w1, w2 = pair
            e_w1, e_w2 = [word_to_vec_map[word] for word in [w1, w2]]

            # Step 2: Compute the mean of e_w1 and e_w2 (≈ 1 line)
            mu = (e_w1 + e_w2) / 2

            # Step 3: Compute the projections of mu over the bias axis and the orthogonal axis (≈ 2 lines)
            mu_B = ((mu @ bias_axis) / np.linalg.norm(bias_axis)**2) * bias_axis
            mu_orth = mu - mu_B

            # Step 4: Use equations (7) and (8) to compute e_w1B and e_w2B (≈2 lines)
            e_w1B = ((e_w1 @ bias_axis) / np.linalg.norm(bias_axis)**2) * bias_axis
            e_w2B = ((e_w2 @ bias_axis) / np.linalg.norm(bias_axis)**2) * bias_axis

            # Step 5: Adjust the Bias part of e_w1B and e_w2B using the formulas (9) and (10) given above (≈2 lines)
            corrected_e_w1B = np.sqrt(1 - np.linalg.norm(mu_orth)**2) * ((e_w1B - mu_B)/np.linalg.norm(e_w1B - mu_B)**2)
            corrected_e_w2B = np.sqrt(1 - np.linalg.norm(mu_orth)**2) * ((e_w2B - mu_B)/np.linalg.norm(e_w2B - mu_B)**2)

            # Step 6: Debias by equalizing e1 and e2 to the sum of their corrected projections (≈2 lines)
            e1 = corrected_e_w1B + mu_orth
            e2 = corrected_e_w2B + mu_orth

            ### END CODE HERE ###

            return e1, e2

        '''
        cosine similarities before equalizing:
        cosine_similarity(word_to_vec_map["man"], gender) =  -0.11711095765336832
        cosine_similarity(word_to_vec_map["woman"], gender) =  0.35666618846270376

        cosine similarities after equalizing:
        cosine_similarity(e1, gender) =  -0.7174005619986723
        cosine_similarity(e2, gender) =  0.7174005619986722
        '''
        ```

    - these debiasing algs are very helpful for reducing bias, but aren’t perfect and don’t eliminate all traces of bias
    - for example, one weakness of this implementation was that the bias direction g was defined using only the pair of words woman and man
    - if g were defined by computing g1, g2, g3 and averaging, a better estimate of the gender dimension in the 50D word embedding space would be obtained