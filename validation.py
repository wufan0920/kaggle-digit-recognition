class validator(object):
    def __init__(self, data, label, shape):
        self.accuracy = []
        self.candidate_result = 0
        self.data = data 
        self.label = label 
        self.weight_shape = shape 

    def validate(self, weight):
        old = weight.shape
        weight.shape = self.weight_shape
        predict = (weight.dot(self.data)).argmax(0)
        accuracy = (predict == self.label.flatten())
        res = accuracy.mean()
        weight.shape = old
        if len(self.accuracy) == 0:
            self.accuracy.append(res)
        elif res >= self.accuracy[0]:
            del self.accuracy[:]
            self.accuracy.append(res)
            self.candidate_result = weight
        else:
            self.accuracy.append(res)

    def early_stop(self):
        return True if len(self.accuracy) == 20 else False
        
    def get_result(self):
        return self.candidate_result

    def get_accuracy(self):
        return self.accuracy[0]

