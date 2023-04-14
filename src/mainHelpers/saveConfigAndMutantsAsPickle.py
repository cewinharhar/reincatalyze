import pickle
def saveConfigAndMutantsAsPickle(config, mutants):
    with open("/home/cewinharhar/GITHUB/reincatalyze/log/config.pkl", "wb") as c:
        pickle.dump(config, file = c)
        c.close()
    with open("/home/cewinharhar/GITHUB/reincatalyze/log/mutants.pkl", "wb") as m:
        pickle.dump(mutants, file = m)
        m.close()