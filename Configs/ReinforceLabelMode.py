from Configs.BaseConfig import Config
import enum

class ReinforcementLabelMode(enum.IntEnum):
    ALTERATIONS_ONLY = 0
    INPUT_ONLY = 1
    ALTERATIONS_AND_INPUT = 2
    ALTERATIONS_AND_INPUT_UNWEIGHTED = 3
    ALTERATIONS_AND_INPUT_TEACHER_FORCING = 4
    ALTERATIONS_AND_INPUT_UNWEIGHTED_TEACHER_FORCING = 5


class IncorrectValueHeadsPolicy(enum.IntEnum):
    DO_NOTHING = 0
    IGNORE_SAMPLE = 1
    INPUT_HEAD_ONLY_SECTION_A = 2
    INPUT_HEAD_ONLY_SECTION_C = 3
    INPUT_HEAD_ONLY_SECTION_A_AND_C = 4
    ALTERATION_HEAD_ONLY_SECTION_A = 5
    ALTERATION_HEAD_ONLY_SECTION_C = 6
    ALTERATION_HEAD_ONLY_SECTION_A_AND_C = 7


class ReinforcementLearningConfig(Config):
    class AdvantageSectionWeights:

        def __init__(self, A=1, B=1, C=1):
            self.A = A
            self.B = B
            self.C = C

    def __init__(self, labelMode=ReinforcementLabelMode.ALTERATIONS_AND_INPUT,
                 inputIncorrectValueHeadsPolicy=IncorrectValueHeadsPolicy.IGNORE_SAMPLE,
                 alterationIncorrectValueHeadsPolicy=IncorrectValueHeadsPolicy.IGNORE_SAMPLE,
                 sectionWeightA=1, sectionWeightB=1, sectionWeightC=1, teacherForcing=True):
        self.labelMode = labelMode
        self.teacherForcing = teacherForcing
        self.inputIncorrectValueHeadsPolicy = inputIncorrectValueHeadsPolicy
        self.alterationIncorrectValueHeadsPolicy = alterationIncorrectValueHeadsPolicy
        self.advantageSectionWeights = self.AdvantageSectionWeights(sectionWeightA, sectionWeightB, sectionWeightC)

    def __str__(self):
        return "A: {}, B: {}, C: {}".format(self.advantageSectionWeights.A, self.advantageSectionWeights.B,
                                            self.advantageSectionWeights.C)
