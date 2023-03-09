# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# In addition to the legal release guidance under MIT please note in this file
# inspired by https://github.com/facebookresearch/SentEval/blob/master/examples/infersent.py 
# that portions of the code are covered by this license: https://github.com/facebookresearch/SentEval/blob/master/LICENSE
import torch
import logging

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s",
    level=logging.INFO,
    filename="evaluation.log",
)

from gensen.encoder import GenSen, GenSenEnsemble

# Set PATHs
PATH_TO_DATA = 'data/'

import senteval

# set gpu device
torch.cuda.set_device(0)


def prepare(params, samples):
    logging.info('Preparing task : %s ' % (params.current_task))
    vocab = set()
    for sample in samples:
        if params.current_task != 'TREC':
            sample = ' '.join(sample).lower().split()
        else:
            sample = ' '.join(sample).split()
        for word in sample:
            if word not in vocab:
                vocab.add(word)

    vocab.add('<s>')
    vocab.add('<pad>')
    vocab.add('<unk>')
    vocab.add('</s>')
    # If you want to turn off vocab expansion just comment out the below line.
    params['gensen'].vocab_expansion(vocab)


def batcher(params, batch):
    # batch contains list of words
    max_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'ImageCaptionRetrieval']
    if params.current_task in max_tasks:
        strategy = 'max'
    else:
        strategy = 'last'

    sentences = [' '.join(s).lower() for s in batch]
    _, embeddings = params['gensen'].get_representation(
        sentences, pool=strategy, normalize=True, add_start_end=True)

    return embeddings


"""
Evaluation of trained model on Transfer Tasks (SentEval)
"""

transfer_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'SST5', 'TREC', 'SICKRelatedness', 'MPRC']

params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10,
                   'classifier': {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                  'tenacity': 5, 'epoch_size': 4}}

if __name__ == "__main__":
    gensen_1 = GenSen(
        model_folder='',
        trainable=True,
        use_cuda=True
    )

    # gensen_2 = GenSen(
    #     model_folder='',
    #     trainable=True,
    #     use_cuda=True
    # )

    # gensen = GenSenEnsemble(gensen_1, gensen_2)

    params_senteval['gensen'] = gensen_1
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    results_transfer = se.eval(transfer_tasks)

    logging.info('--------------------------------------------')
    logging.info('Table 2 of Our Paper : ')
    logging.info('--------------------------------------------')
    logging.info(
        'MR                [Dev:%.1f/Test:%.1f]' % (results_transfer['MR']['devacc'], results_transfer['MR']['acc']))
    logging.info(
        'CR                [Dev:%.1f/Test:%.1f]' % (results_transfer['CR']['devacc'], results_transfer['CR']['acc']))
    logging.info('SUBJ              [Dev:%.1f/Test:%.1f]' % (
    results_transfer['SUBJ']['devacc'], results_transfer['SUBJ']['acc']))
    logging.info('MPQA              [Dev:%.1f/Test:%.1f]' % (
    results_transfer['MPQA']['devacc'], results_transfer['MPQA']['acc']))
    logging.info('SST2              [Dev:%.1f/Test:%.1f]' % (
    results_transfer['SST2']['devacc'], results_transfer['SST2']['acc']))
    logging.info('SST5              [Dev:%.1f/Test:%.1f]' % (
    results_transfer['SST5']['devacc'], results_transfer['SST5']['acc']))
    logging.info('TREC              [Dev:%.1f/Test:%.1f]' % (
    results_transfer['TREC']['devacc'], results_transfer['TREC']['acc']))
    logging.info('MRPC              [Dev:%.1f/TestAcc:%.1f/TestF1:%.1f]' % (
    results_transfer['MRPC']['devacc'], results_transfer['MRPC']['acc'], results_transfer['MRPC']['f1']))
    logging.info('SICKRelatedness   [Dev:%.3f/Test:%.3f]' % (
    results_transfer['SICKRelatedness']['devpearson'], results_transfer['SICKRelatedness']['pearson']))
    logging.info('--------------------------------------------')
