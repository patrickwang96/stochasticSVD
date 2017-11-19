
import pickle

if __name__ == '__main__':
    filePath = 'tmp_dataset.txt'
    with open(filePath, 'r') as f:
        raw_dataset = f.readlines()

    dataset1 = map(lambda i: i.strip().split(','), dataset)

    dataset2 = map(lambda i: (int(i[0]), i[1:]), dataset1)

    del dataset1

    word_dict = dict()

    for document in dataset2:
        document = document[1]

        for token in document:
            count = word_dict.get(token, 0)
            count = count + 1
            word_dict[token] = count

    print('Finished calculating...')
    pickle.dump(word_dict, open('word_dict_dump.p', 'wb'))