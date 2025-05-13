from torch import nn, div, matmul, transpose, squeeze
from torch.nn.functional import normalize
from pytorch_metric_learning import losses


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, feature_vectors, labels):
        feature_vectors_normalised = normalize(feature_vectors, p=2, dim=1)
        logits = div(
            matmul(feature_vectors_normalised,
                   transpose(feature_vectors_normalised, dim0=0, dim1=1)),
            self.temperature)

        return losses.NTXentLoss(temperature=0.07)(logits, squeeze(labels))
