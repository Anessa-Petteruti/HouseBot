import os
import numpy as np
import pandas as pd
#import ConceptNet

def process_object_labels():
    """
    Performs any necessary processing on object labels outputted by Detectron2.
    Detectron2 is run in a separate GoogleColab, dataset of images and labels are saved in repo.
    """
    pass

def find_actions():
    """
    Calls ConceptNet to output a set of actions for input object labels.
    """
    pass

def train():
    """
    """
    pass

def test():
    """
    Compare ConceptNet actions with AI2Thor verbs.
    """
    pass

def f1_score():
    """
    Evaluation metric that takes both precision and recall into account.
    """
    pass

def mean_avg_prec():
    """
    Calculates Mean Average Precision (MAP), the mean precision across each result.
    """
    pass

def main():
    # Will need to pass things in to these functions here...
    process_object_labels()


if __name__ == "__main__":
    main()