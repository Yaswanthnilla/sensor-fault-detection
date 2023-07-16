from sklearn.metrics import f1_score,precision_score,recall_score
from sensor.entity.artifact_entity import ClassificationMetricArtifact
from sensor.exception import SensorException
import sys,os

def get_classification_score(y_true,y_pred)->ClassificationMetricArtifact:
    try:
        model_f1_score = float(f1_score(y_true, y_pred))
        model_recall_score = float(recall_score(y_true, y_pred))
        model_precision_score= float(precision_score(y_true,y_pred))

        classification_metric_artifact=ClassificationMetricArtifact(f1_score=model_f1_score,
                                                                    recall_score=model_recall_score,
                                                                    precision_score=model_precision_score)
        
        return classification_metric_artifact

    except Exception as e:
        raise SensorException(e,sys)


def to_dict(classification_metric_artifact:ClassificationMetricArtifact):
    return classification_metric_artifact.__dict__