import pypendula_n3
from tqdm import trange


for run in trange(12):
    ics = pypendula_n3.gen_rand_ics()
    pypendula_n3.main(ics=ics, run=run, verbose=False)
