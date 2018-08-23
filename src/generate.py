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

        true_list.append(test_data[n_steps + dummy_iters + i])
        pred_list.append(indices_val[predicted_grammar])
    
    print(true_list)
    print(pred_list)

    return(true_listpred_list)