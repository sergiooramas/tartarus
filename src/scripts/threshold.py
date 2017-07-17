from sklearn.metrics import pairwise, precision_recall_curve, average_precision_score, roc_curve
import numpy as np
import pandas as pd
import common
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_threshold(model_id):
    trained_models = pd.read_csv(common.DEFAULT_TRAINED_MODELS_FILE, sep='\t')
    model_config = trained_models[trained_models["model_id"] == model_id]
    if model_config.empty:
        raise ValueError("Can't find the model %s in %s" %
                         (model_id, common.DEFAULT_TRAINED_MODELS_FILE))
    model_config = model_config.to_dict(orient="list")
    model_settings=eval(model_config['dataset_settings'][0])

    Y_test = np.load(common.DATASETS_DIR+'/item_factors_test_%s_%s_%s.npy' % (model_settings['fact'],model_settings['dim'],model_settings['dataset']))
    Y_pred = np.load(common.FACTORS_DIR+'/factors_%s.npy' % model_id)

    good_scores = Y_pred[Y_test==1]
    th = good_scores.mean()
    std = good_scores.std()
    print 'Mean th',th
    print 'Std',std

    p, r, thresholds = precision_recall_curve(Y_test.flatten(), Y_pred.flatten())
    f = np.nan_to_num((2 * (p*r) / (p+r)) * (p>r))
    print f
    max_f = np.argmax(f)
    fth = thresholds[max_f]
    print f[max_f],p[max_f],r[max_f]
    print 'F th %.2f' % fth
    plt.plot(r, p, 
             label='Precision-recall curve of class {0}')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of Precision-Recall curve to multi-class')
    plt.savefig("pr_curve.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluates the model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(dest="model_id",
                        type=str,
                        help='Identifier of the Model to evaluate')

    args = parser.parse_args()
    get_threshold(args.model_id)

