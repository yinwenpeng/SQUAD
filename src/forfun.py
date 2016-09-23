

def refine_text(sequence):
    seq=sequence.replace('. ', ' . ')
    if seq[-1]=='.':
        seq=seq[:-1]+' .'
    return seq


if __name__ == '__main__':
    import operator
    x = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}
    sorted_x = sorted(x.items(), key=operator.itemgetter(1), reverse=True)
    print sorted_x
    for key, value in sorted_x:
        print key, value




