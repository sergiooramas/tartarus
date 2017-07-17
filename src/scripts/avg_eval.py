import common
import numpy as np
import glob
import argparse

def evaluate(model_id, stepsize="10000", measure="map"):
    files = glob.glob(common.DATA_DIR+"/eval/%s-%s/%s_*.txt" % (model_id,stepsize,measure))
    print len(files)
    for n in range(1,len(files)):
        sum = 0
        for file in files[:n]:
            sum += float(open(file).read())
        avg = sum/n
    return avg, stepsize*len(files)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluates the model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(dest="model_id",
                        type=str,
                        help='Identifier of the Model to evaluate')
    parser.add_argument('-m',
                        '--measure',
                        dest="measure",
                        help='Evaluation measure',
                        default="map")
    parser.add_argument('-s',
                        '--stepsize',
                        dest="stepsize",
                        help='Evaluation measure',
                        default=10000)

    args = parser.parse_args()
    res, n_users = evaluate(args.model_id,args.stepsize,args.measure)
    print "%.5f" % res, n_users
    fw=open(common.DATA_DIR+'/results/eval_results.txt','a')
    fw.write("\n")
    fw.write("%s - %s users\n" % (args.model_id,n_users))
    fw.write("%s %.5f" % (args.measure,res))
    fw.write("\n")

