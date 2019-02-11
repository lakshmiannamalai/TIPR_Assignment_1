import bayes as bayes
import nn as nn
import lsh as lsh
import projections as RP

if __name__ == '__main__':
    print('Welcome to the world of high and low dimensions!')
    # The entire code should be able to run from this file!
    import argparse
    
    parser = argparse.ArgumentParser(description='Create a ArcHydro schema')
    parser.add_argument('--datafile', required=True,
                        help='path to datafile')
    parser.add_argument('--labelfile', required=True,
                        help='path to labelfile')
    parser.add_argument('--dataset', required=True,
                        help='dataset name: dolphins / pubmed / twitter')
    args = parser.parse_args()
    if(args.dataset != 'dolphins' and args.dataset != 'pubmed' and args.dataset != 'twitter'):
        print("The entered dataset name is incorrect. Please enter dataset name (dophins / pubmed / twitter)")
    else:
        [Accuracy, F1macro, F1micro] = bayes.bayesClassifier(args.datafile,args.labelfile,args.dataset)
        [Accuracy, F1macro, F1micro] = nn.NearestNeighbor(args.datafile,args.labelfile,args.dataset)
        [Accuracy, F1macro, F1micro] = lsh.LSH(args.datafile,args.labelfile,args.dataset)
        #RP.RandomProjection(args.datafile,args.labelfile,args.dataset)