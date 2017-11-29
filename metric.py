import numpy as np


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


    def update(self, preds):

        assert len(preds)==12,'output format error'


        for i in range(len(preds)):

            pred =preds[i].asnumpy()

            self.sum_metric[i] += pred.sum()/pred.shape[0]/2
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