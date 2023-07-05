# region imports


import json
import numpy as np
import os

import random

import openai

import time



def preprocess(filename,delimiter=',',format='conll'):
    """
    if filename.endswith('.csv'):
        with open(filename, 'rt') as f:
            data = csv.reader(f,delimiter=delimiter)
            data=list(data)

    else:
       """
    with open(filename) as myfile:
            data = myfile.readlines()
            data = [i.rstrip('\n') for i in data]

    if format=='edgar':
                data = [i.rsplit(delimiter,1) for i in data]
    elif format=='conll' or format=='others':
                data = [i.split(delimiter) for i in data]



    if format=='conll':
            for i in data:
                if i != [''] and i!=[]:
                    del i[1]
                    del i[1]  # delete the middle 2 columns from the data
    for i in range(0, len(data)):
                if data[i] == [''] or data[i]==[]:
                    data[i] = ["", "O"]


    return data


def endofphrase(prev, current):#if the previous word is the last word of a NE phrase, then returns true
    answer=False
    if prev.startswith("B") and current.startswith("B"):
        answer=True
    if prev.startswith("B") and current.startswith("O"):
        answer=True
    if prev.startswith("I") and current.startswith("B"):
        answer=True
    if prev.startswith("I") and current.startswith("O"):
        answer=True
    if prev!="O" and current!="O" and prev[2:]!=current[2:]:
        answer=True
    return answer




def startofphrase(prev, current):  #if the current word is the first word of a NE phrase, then returns true
    answer=False
    if current.startswith("B"):
        answer=True
    if prev.startswith("O") and current.startswith("I"):
        answer=True
    if prev!="O" and current!="O" and prev[2:]!=current[2:]:
        answer=True
    return answer




def word2sents(data_string_list):
    data = list()
    X = list()
    Y = list()
    for data_string in data_string_list:

        if data_string == ['', 'O']:
            if X == [['-DOCSTART-']]:
                X = list()
                Y = list()
                continue

            data.append((X, Y))
            X = list()
            Y = list()
        else:

            X.append(data_string[:-1])
            Y.append(data_string[-1])

    if len(X) > 0:
        data.append((X, Y))

    data = [x for x in data if len(x) != 0]

    return data



prompt_start={'ORG': f"Extract organization entities in the following sentence. " \
                 f"Organization entities are limited to named corporate, governmental, or other organizational entities."
                     f" Surround the extracted entities by @@ and ##."
              f" Below are some examples.\n" \
                 f"Input: EU rejects German call to boycott British lamb .\n"
                 f"Output: @@EU## rejects German call to boycott British lamb .\n"
              f"Input: Peter Blackburn\n"
              f"Output: Peter Blackburn\n"
              f"Input: BRUSSELS 1996-08-22\n"
              f"Output: BRUSSELS 1996-08-22\n",

              'LOC':f"Extract location entities in the following sentence. " \
                 f"Location entities are the name of politically or geographically defined locations such as cities, provinces, countries, international regions, bodies of water, mountains, etc."
             f" Surround the extracted entities by @@ and ##."
              f" Below are some examples.\n" \
                 f"Input: EU rejects German call to boycott British lamb .\n"
                 f"Output: EU rejects German call to boycott British lamb .\n"
              f"Input: Peter Blackburn\n"
              f"Output: Peter Blackburn\n"
              f"Input: BRUSSELS 1996-08-22\n"
              f"Output: @@BRUSSELS## 1996-08-22\n",

              'PER': f"Extract person entities in the following sentence. " \
                     f"Person entities are named persons or family."
                       f" Surround the extracted entities by @@ and ##."
                     f" Below are some examples.\n" \
                     f"Input: EU rejects German call to boycott British lamb .\n"
                     f"Output: EU rejects German call to boycott British lamb .\n"
                     f"Input: Peter Blackburn\n"
                     f"Output: @@Peter Blackburn##\n"
                     f"Input: BRUSSELS 1996-08-22\n"
                     f"Output: BRUSSELS 1996-08-22\n",

              'MISC': f"Extract miscellaneous entities in the following sentence. " \
                     f"Miscellaneous entities include events, nationalities, products and works of art."
                    f" Surround the extracted entities by @@ and ##."
                     f" Below are some examples.\n" \
                     f"Input: EU rejects German call to boycott British lamb .\n"
                     f"Output: EU rejects @@German## call to boycott @@British## lamb .\n"
                     f"Input: Peter Blackburn\n"
                     f"Output: Peter Blackburn\n"
                     f"Input: BRUSSELS 1996-08-22\n"
                     f"Output: BRUSSELS 1996-08-22\n"
              }



def mrc2prompt(testdata,  example_idx, traindata,data_name="CONLL",):
    print("mrc2prompt ...")

    def get_example(index, l):
        exampel_prompt = ""

        for idx_ in index:

            context = traindata[idx_][0]
            context = [item for sublist in context for item in sublist]
            context = ' '.join(context)

            truth_labels = traindata[idx_][1]


            context_list = context.strip().split()

            prev_tag = 'O'
            for i in range(0, len(truth_labels) + 1):

                if i == 0:
                    prev_tag = 'O'
                else:
                    prev_tag = truth_labels[i - 1]
                if i == len(truth_labels):
                    current_tag = 'O'
                else:
                    current_tag = truth_labels[i]

                if startofphrase(prev_tag, current_tag):
                    if current_tag.endswith(l):
                        context_list[i] = "@@" + context_list[i]
                if endofphrase(prev_tag, current_tag):
                    if prev_tag.endswith(l):
                        context_list[i - 1] = context_list[i - 1] + '##'

            exampel_prompt += f"Input: {context}\n"

            exampel_prompt += f"Output: {' '.join(context_list)}\n"
            #exampel_prompt += '\n'
        return exampel_prompt


    prompts_of_label = []
    labels = ['ORG','LOC','PER','MISC']
    for label in labels:

        prompt=prompt_start[label]

        prompt += get_example(index=example_idx, l=label)

        #prompt += f"Input: {context}\nOutput:"

        # print(prompt)
        prompts_of_label.append(prompt)



    results = []
    for item_idx in range(len(testdata)):

        prompts_of_sentence=[]

        item_ = testdata[item_idx][0]
        context = [item for sublist in item_ for item in sublist]
        context = ' '.join(context)

        for prompt in prompts_of_label:
            final_prompt =prompt+ f"Input: {context}\nOutput:"
            prompts_of_sentence.append(final_prompt)


        results.append(prompts_of_sentence)

    return results

import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--openai_api_key", type=str,
                        help="openai key")
    parser.add_argument("--train", type=str,
                        help="path to train file")
    parser.add_argument("--test", type=str,
                        help="path to test file")
    parser.add_argument("--output",
                        help="path of output")
    args = parser.parse_args()

    openai.api_key = "put api key here"

    train= "put path to train file here"
    test= "put path to test file here"
    output_dir = "put path to output here"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)



    data_string_list = preprocess(train, delimiter=' ', format='conll')
    traindata = word2sents(data_string_list)

    data_string_list = preprocess(test, delimiter=' ', format='conll')
    testdata = word2sents(data_string_list)

    testdata=testdata[:100]

    for s in [0,1,2]:


        idx=list(range(0,len(traindata)))

        #manually remove these indexes (0,1,2), because they are ( cherry picked) used as positive examples above.
        idx.remove(0)
        idx.remove(1)
        idx.remove(2)

        random.seed(s)
        random.shuffle(idx)  #set seed to 0 and shuffle once.


        #for CoNLL, K=61+3, it takes about 2600 tokens


        for K in [1,5,13,29,61]:   #because we already have 3 examples in our above, adding these extra K examples will make result in 4,8,16,32-shot learning

            prompts = mrc2prompt(testdata=testdata, data_name='CONLL', example_idx=idx[:K],
                                 traindata=traindata)

            # ==============================================================================================================================



            all_predictions = {}
            labels = ['ORG', 'PER', 'LOC', 'MISC']

            for i in range(0, len(prompts)):

                sent = prompts[i]

                predictions = {}

                for j in range(0, len(sent)):

                    print(i, j)
                    prompt = sent[j]
                    done = False
                    while done == False:
                        try:

                            response = openai.ChatCompletion.create(

                                model="gpt-4",  #we can use gpt-4 32K tokens, but not necessary at this stage

                                messages=[
                                    {"role": "user", "content": prompt}
                                ],
                                temperature=0,
                                stop=['\n', 'Input:']

                            )

                            done = True
                            #total_tokens += response['usage']['total_tokens']

                            # print the completion
                            # print(response['choices'][0]['message']['content'].strip(" \n"))
                            prediction = response['choices'][0]['message']['content'].strip(" \n")

                            l = labels[j]
                            predictions[l] = prediction


                        except openai.error.InvalidRequestError:
                            #exceed.append((i, j))
                            #print('Too Long')

                            done = True

                            l = labels[j]
                            predictions[l] = None



                        except openai.error.RateLimitError:
                            #print('too much request')
                            time.sleep(5)
                        #except:

                        except Exception as e:
                            print(e)

                            #print('any other errors')
                            #time.sleep(5)

                all_predictions[i] = predictions


            save_path=os.path.join(output_dir,"seed%s"%s,"%s_shot_pred.json"%(K+3))

            with open(save_path, 'w') as fp:
                json.dump(all_predictions, fp, indent=2)



if __name__ == "__main__":
    main()
