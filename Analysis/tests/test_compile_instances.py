import os
import unittest

from classes import Constraint
from compile_instances import *
from test_functions import *

from pandas._testing import assert_frame_equal

preposition_list = StudyInfo.preposition_list

os.chdir("..")


class Test(unittest.TestCase):
    """Summary
    """

    def test_sv_collection(self):
        study_info = StudyInfo("2019 study")

        svcollection = SemanticCollection(study_info)

        for i in svcollection.instance_list:
            self.assertTrue(i.preposition in preposition_list or i.preposition == 'none')
            self.assertIsInstance(i, Instance)

        #
        self.assertTrue(all(x in preposition_list + ['none'] for x in svcollection.get_used_prepositions()))

        filename = svcollection.study_info.data_folder + "/" + svcollection.study_info.sem_annotations_name

        with open(filename, "r") as f:
            reader = csv.reader(f)  # create a 'csv reader' from the file object
            annotationlist = list(reader)  # create a list from the reader

            annotationlist.pop(0)  # removes first line of data list which is headings

            for clean_annotation in annotationlist:
                an_id, clean_user_id, task, scene, prepositions, figure, ground, time = SemanticAnnotation.retrieve_from_data_row(
                    clean_annotation)
                removed_empty = []
                for p in prepositions:
                    if p != '':
                        removed_empty.append(p)
                self.assertTrue(all(p in preposition_list + ['none'] for p in removed_empty))

        # test outputs
        svcollection.write_config_ratios()

        for preposition in svcollection.get_used_prepositions():
            print(preposition)

            new_df, original_df = generate_dataframes_to_compare(
                svcollection.study_info.config_ratio_csv(svcollection.filetag, preposition))
            # print(new_df)
            # print(original_df)
            assert_frame_equal(new_df, original_df)

    # @unittest.skip
    def test_comp_collection(self):
        study_info = StudyInfo("2019 study")
        compcollection = ComparativeCollection(study_info)

        for i in compcollection.instance_list:
            self.assertIsInstance(i, CompInstance)
            self.assertTrue(i.preposition in preposition_list or i.preposition == 'none')
            self.assertIsInstance(i.possible_figures, list)

        compcollection.get_and_write_constraints()
        constraints = Constraint.read_from_csv(study_info.constraint_csv)
        new_constraint_df, original_df = generate_dataframes_to_compare(study_info.constraint_csv)

        # Check theres no repetitions
        for preposition in study_info.preposition_list:
            for con in constraints[preposition]:
                row_check = (new_constraint_df['scene'] == con.scene) & (
                        new_constraint_df['preposition'] == con.preposition) & (
                                    new_constraint_df['ground'] == con.ground) & (new_constraint_df['f1'] == con.f1) & (
                                    new_constraint_df['f2'] == con.f2)

                single_constraint_df = new_constraint_df.loc[row_check, :]

                self.assertEqual(len(single_constraint_df), 1)

                # Check this instance
                if con.scene == "compsva25" and con.preposition == "in" and con.ground == "bin" and con.f1 == "picture" and con.f2 == "table":
                    self.assertEqual(con.weight, 1)
                    self.assertAlmostEqual(con.lhs["support"], 1.37388140178845)
                    self.assertAlmostEqual(con.rhs["support"], -0.6282480394372745)
                    print("Instance found")


if __name__ == "__main__":
    unittest.main()
