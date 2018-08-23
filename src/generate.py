'''
Author:     Alex Kim
Project:    DeepMelodies
File:       src/generate.py
Purpose:    generate new grammars with trained model

'''

import numpy as np

def generate(model, test_data, example_test_generator, indices_val, n_steps, n_predict, dummy_iters):

    for i in range(dummy_iters):
        dummy = next(example_test_generator.generate())

    true_list = []
    pred_list = []
    for i in range(n_predict):
        data = next(example_test_generator.generate())
        prediction = model.predict(data[0])
        predicted_grammar = np.argmax(prediction[:, n_steps - 1, :])

        if i == 0: # Add first few notes
            true_list.extend([indices_val[i] for i in data[0][0]]) 
            pred_list.extend([indices_val[i] for i in data[0][0]])
        else:
            true_list.append(indices_val[data[0][0][-1]])
            # If prediction not in vocabulary, subtract to adjust for >1 batches
            while predicted_grammar not in indices_val: 
                # print("shape: %d, predicted: %d" % (prediction.shape[2], predicted_grammar))
                predicted_grammar -= prediction.shape[2]
                
            pred_list.append(indices_val[predicted_grammar])

    return(true_list, pred_list)