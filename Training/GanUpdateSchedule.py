class GenDiscUpdateSchedule:

    def __init__(self, saveGen, saveDisc):
        self.saveDisc = saveDisc
        self.saveGen = saveGen

    def getGenDiscUpdates(self, stepCounter, epochCounter, genOptimWraper, discOptimWraper):
        raise NotImplementedError()


class AlwaysUpdateSchedule(GenDiscUpdateSchedule):

    def __init__(self):
        super().__init__(True, True)

    def getGenDiscUpdates(self, stepCounter, epochCounter, genOptimWraper, discOptimWraper):
        return True, True

    def __repr__(self):
        return 'Always Update Both'


class NeverUpdateSchedule(GenDiscUpdateSchedule):

    def __init__(self):
        super().__init__(False, False)

    def getGenDiscUpdates(self, stepCounter, epochCounter, genOptimWraper, discOptimWraper):
        return False, False

    def __repr__(self):
        return 'Never Update Any'


class OnlyUpdateDisc(GenDiscUpdateSchedule):

    def __init__(self):
        super().__init__(saveGen=False, saveDisc=True)

    def getGenDiscUpdates(self, stepCounter, epochCounter, genOptimWraper, discOptimWraper):
        return True, False

    def __repr__(self):
        return 'Only Update Disc'


class OnlyUpdateGen(GenDiscUpdateSchedule):

    def __init__(self):
        super().__init__(saveGen=True, saveDisc=False)

    def getGenDiscUpdates(self, stepCounter, epochCounter, genOptimWraper, discOptimWraper):
        return False, True

    def __repr__(self):
        return 'Only Update Generator'


class AlternateSequence(GenDiscUpdateSchedule):

    def __init__(self, gen2diskUpdateMod=2):
        super().__init__(saveGen=True, saveDisc=True)
        self.gen2diskUpdateMod = gen2diskUpdateMod

    def getGenDiscUpdates(self, stepCounter, epochCounter, genOptimWraper, discOptimWraper):
        if (stepCounter % self.gen2diskUpdateMod == 0):
            return False, True
        return True, False

    def __repr__(self):
        return 'Alternate Updates. GenUpdateMod: {}'.format(self.gen2diskUpdateMod)


class AlternateOverEpochsUpdateSchedule(GenDiscUpdateSchedule):

    def __init__(self):
        super().__init__(saveGen=True, saveDisc=True)

    def getGenDiscUpdates(self, stepCounter, epochCounter, genOptimWraper, discOptimWraper):
        if (epochCounter % 2 == 0):
            return False, True
        return True, False

    def __repr__(self):
        return 'Alternate between updates between epochs'


class GeneratorSlowStart(GenDiscUpdateSchedule):

    def __init__(self, genStartStep):
        super().__init__(saveGen=True, saveDisc=True)
        self.genStartStep = genStartStep

    def getGenDiscUpdates(self, stepCounter, epochCounter, genOptimWraper, discOptimWraper):
        if (stepCounter < self.genStartStep):
            return False, True
        return True, True

    def __repr__(self):
        return 'Generator Slow Start. GenStartStep: {}'.format(self.genStartStep)
