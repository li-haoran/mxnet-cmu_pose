import numpy as np

def check_label_shapes(labels, preds, shape=0):
    if shape == 0:
        label_shape, pred_shape = len(labels), len(preds)
    else:
        label_shape, pred_shape = labels.shape, preds.shape

    if label_shape != pred_shape:
        raise ValueError("Shape of labels {} does not match shape of "
                         "predictions {}".format(label_shape, pred_shape))

class RMSEWithone(object):
    def __init__(self, stages=12,name='RMSEWithone'):
        self.stages=stages
        self.sum_metric=np.zeros((self.stages,))
        self.num_inst=0
        self.name=name

    def reset(self):
        """Clear the internal statistics to initial state."""
        self.num_inst = 0
        self.sum_metric = np.zeros((self.stages,))


    def update(self, labels, preds):

        assert len(labels)==2,'labels format error'
        assert len(preds)==12,'output format error'


        for i in range(len(preds)):

            label = labels[i%2].asnumpy()
            pred =preds[i].asnumpy()

            check_label_shapes(label, pred)

            self.sum_metric[i] += ((label - pred)**2.0).sum()/pred.shape[0]/2
        self.num_inst += 1

    def get(self):
        """Get the current evaluation result.

        Returns
        -------
        name : str
           Name of the metric.
        value : float
           Value of the evaluation.
        """
        
        if self.num_inst == 0:
            return (self.name, float('nan'))
        else:
            return (self.name, self.sum_metric / self.num_inst)
        
            return (names, values)