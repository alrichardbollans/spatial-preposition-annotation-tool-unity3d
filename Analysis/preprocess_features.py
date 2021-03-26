"""Summary
Contains Features class which, for a given study, reads file of extracted features, removes some features,
gets average location control, standardises values and outputs
"""
from data_import import StudyInfo


def process_all_features(study):
    """Summary
    """

    f = study.feature_processor
    nd = f.standardise_values()
    f.write_new(nd)
    f.write_mean_std()


if __name__ == '__main__':

    process_all_features(StudyInfo("2019 study"))
    process_all_features(StudyInfo("2020 study"))