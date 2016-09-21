

def refine_text(sequence):
    seq=sequence.replace('. ', ' . ')
    if seq[-1]=='.':
        seq=seq[:-1]+' .'
    return seq


if __name__ == '__main__':
    
    lis=['Upper', 'heihei', 'YoHe']
    
    print map(lower, lis)





