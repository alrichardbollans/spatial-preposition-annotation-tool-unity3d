"""Summary
Reads feature value file from unity project and standardises values
Uses "standard score"
Outputs in given directory
"""

import pandas as pd

from classes import BasicInfo


def process_all_features():
    """Summary
    """
    for study in BasicInfo.study_list:
        try:
            f = Features(study)
            nd = f.standardise_values()
            f.write_new(nd)
            f.write_mean_std()
        except Exception as e:
            print("Study features don't exist")
            print(study)
            print(e)


class Features:
    """Summary
    
    Attributes:
        dataset (TYPE): Description
        features_to_remove (TYPE): Description
        human_readable_path (TYPE): Description
        means (TYPE): Description
        output_path (TYPE): Description
        input_feature_csv (TYPE): Description
        stds (TYPE): Description
    """

    # Features given in relation.csv that aren't needed for analysis
    features_to_remove = ["size_ratio", "horizontal_distance_normalized", "shortest_distance_normalized",
                          "location_control_-x", "location_control_-z", "location_control_x",
                          "location_control_z", "figure_volume", "ground_volume",
                          "horizontal_projection_overlap_proportion"]

    def __init__(self, study):
        """Summary
        
        Args:
            study (TYPE): Description
        """
        basic_info = BasicInfo(study)
        # Get path for calculated features
        self.input_feature_csv = basic_info.input_feature_csv
        self.output_path = basic_info.feature_output_csv
        self.human_readable_path = basic_info.human_readable_feature_output_csv

        self.dataset = pd.read_csv(self.input_feature_csv)
        # Remove values where fig=ground

        self.dataset = self.dataset[self.dataset.Figure != self.dataset.Ground]

        # Calculate location control and append column
        location_control_parts = ["location_control_x", "location_control_-x", "location_control_-z",
                                  "location_control_z"]

        self.dataset["location_control"] = self.dataset[location_control_parts].sum(axis=1) / 4

        # Remove unwanted columns
        self.dataset = self.dataset.drop(self.features_to_remove, axis=1)
        # Output cleaned dataset
        self.dataset.to_csv(self.human_readable_path, index=False)
        # Calculate means and standard deviations
        # Can be used to convert standardised values back to real values
        self.means = self.dataset.iloc[:, 3:].mean()
        self.stds = self.dataset.iloc[:, 3:].std()

    def standardise_values(self):
        """Summary
        
        Returns:
            TYPE: Description
        """
        new_dataframe = self.dataset.copy()

        new_dataframe.iloc[:, 3:] = new_dataframe.iloc[:, 3:] - self.means
        new_dataframe.iloc[:, 3:] = new_dataframe.iloc[:, 3:] / (self.stds)

        return new_dataframe

    def write_new(self, dataframe):
        """Summary
        
        Args:
            dataframe (TYPE): Description
        """
        dataframe.to_csv(self.output_path, index=False)

    def write_mean_std(self):
        """Summary
        """
        self.means.to_csv("feature values/feature_means.csv")
        self.stds.to_csv("feature values/feature_stds.csv")

    def convert_standardised_value_to_normal(self, feature, value):
        """Summary
        
        Args:
            feature (TYPE): Description
            value (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        mean = self.means[feature]
        std = self.stds[feature]

        x = (value * std) + mean

        return x

    def convert_normal_value_to_standardised(self, feature, value):
        """Summary
        
        Args:
            feature (TYPE): Description
            value (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        mean = self.means[feature]
        std = self.stds[feature]

        x = (value - mean) / std

        return x

    def get_max_value(self, feature):
        """Summary
        
        Args:
            feature (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        m = self.dataset[feature].max()
        return m

    def get_min_value(self, feature):
        """Summary
        
        Args:
            feature (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        m = self.dataset[feature].min()
        return m

    def convert_standard_df_to_normal(self, df):
        """Summary
        
        Args:
            df (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        # Need to be careful assigning new dataframe variables this way
        # As it returns a reference to the initial dataframe UNLESS copy() is used
        new_dataframe = df.copy()
        for column in new_dataframe:
            mean = self.means[column]
            std = self.stds[column]

            new_dataframe[column] = (new_dataframe[column] * std) + mean

        return new_dataframe


if __name__ == '__main__':
    process_all_features()
